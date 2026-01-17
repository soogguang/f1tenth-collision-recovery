#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from ament_index_python.packages import get_package_share_directory
import rclpy
import time
from rclpy.node import Node
import tensorflow as tf
import numpy as np
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from rclpy.parameter import Parameter
from rcl_interfaces.msg import SetParametersResult
from std_msgs.msg import Bool
from std_msgs.msg import String

# ===================== 기본 튜닝값 =====================
DOWN_SAMPLE = 2
EMA_ALPHA = 0.25    #0.7
STEERING_DEADZONE_RAD = 0.0

CURVE_SPEED_WEIGHT = 0.35  #0.65
V_MIN = 0.0   #2.0
V_MAX = 3.0   #3.0

STEER_ABS_MAX = 0.75   #0.36
# =====================================================

class TLNInference(Node):
    def __init__(self):
        super().__init__('tln_inference')

        # 모델 로드
        pkg_share = get_package_share_directory('ofc')
        self.model_path = str(Path(pkg_share) / 'models' / 'f2_f4_silverstone_7lap.keras')
        self.model = tf.keras.models.load_model(self.model_path, compile=False, safe_mode=False)
        self.get_logger().info("모델 로드 완료 - GPU backend 활성 ")
        self.steer_ema = 0.0

        # ---------- 파라미터 선언 ----------
        self.declare_parameter("drive_publish_enabled", False)

        self.declare_parameters('', [
            ('scan_topic', '/scan'),
            ('drive_topic', '/drive'),

            ('down_sample', DOWN_SAMPLE),
            ('ema_alpha', EMA_ALPHA),
            ('steering_deadzone_rad', STEERING_DEADZONE_RAD),

            # 런타임 좌우반전 스위치(입력 뒤집고, 조향 부호 반전)
            ('reverse_direction', False),

            ('curve_speed_weight', CURVE_SPEED_WEIGHT),
            ('v_min', V_MIN),
            ('v_max', V_MAX),
            ('steer_abs_max', STEER_ABS_MAX),

            # 매핑 모드 스위치
            ('mapping', False),

            # 라이다 값 클램핑 상한 (기본 10.0 m)
            ('clamp_range_max_m', 10.0),
        ])

        # 값 로드
        self.drive_publish_enabled = bool(self.get_parameter("drive_publish_enabled").value)
        self.scan_topic            = str(self.get_parameter('scan_topic').value)
        self.drive_topic           = str(self.get_parameter('drive_topic').value)
        self.down_sample           = int(self.get_parameter('down_sample').value)
        self.ema_alpha             = float(self.get_parameter('ema_alpha').value)
        self.steering_deadzone_rad = float(self.get_parameter('steering_deadzone_rad').value)

        self.reverse_direction     = bool(self.get_parameter('reverse_direction').value)

        self.curve_speed_weight    = float(self.get_parameter('curve_speed_weight').value)
        self.v_min                 = float(self.get_parameter('v_min').value)
        self.v_max                 = float(self.get_parameter('v_max').value)
        self.steer_abs_max         = float(self.get_parameter('steer_abs_max').value)

        self.mapping               = bool(self.get_parameter('mapping').value)

        # 클램핑 상한 파라미터
        self.clamp_range_max_m     = float(self.get_parameter('clamp_range_max_m').value)

        # 변경 콜백 등록
        self.add_on_set_parameters_callback(self._on_params)
        # -----------------------------------

        # 구독/퍼블리셔
        self.sub = self.create_subscription(LaserScan, self.scan_topic, self.scan_cb, 5)
        self.pub = self.create_publisher(AckermannDriveStamped, self.drive_topic, 5)

        # ---------------------- 추가 부분 ----------------------
        # 새 충돌 상태 머신 변수
        self.collision = False
        self.brake_phase = False
        self.reverse_phase = False
        #TLN에서 컨트롤러로 변경 시 제어가 불가능할 때 사용
        #self.pf_stable_ok = False
        #self.state = "NORMAL"

        self.collision_start_time = None
        self.brake_hold_duration = 0.1   # 정지 유지 시간
        self.reverse_duration = 2.0     # 후진 시간

        # 충돌 메시지 구독
        self.backward_done_pub = self.create_publisher(Bool, "/tln/backward_done", 5)
        self.create_subscription(Bool, "/collision_detected", self.collision_cb, 5)
        #TLN에서 컨트롤러로 변경 시 제어가 불가능할 때 사용
        #self.create_subscription(String, "/state", self.state_cb, 5)
        #self.create_subscription(Bool, "/pf/stable_ok", self.pf_stable_ok_cb, 5)
        # -------------------------------------------------------

    def dnn_output(self, arr):
        if arr is None:
            return 0., 0.
        return self.model(arr, training=False).numpy()[0]

    def make_hokuyo_scan(self, arr):
        # 1080 → 1081 보정(모델 입력 정합용)
        if arr.shape[0] == 1080:
            arr = np.append(arr, arr[-1])
        return arr

    def linear_map(self, x, x_min, x_max, y_min, y_max):
        return (x - x_min) / (x_max - x_min) * (y_max - y_min) + y_min

    #TLN에서 컨트롤러로 변경 시 제어가 불가능할 때 사용
    # def state_cb(self, msg):
    #     self.state = msg.data.upper()   
    # def pf_stable_ok_cb(self, msg):
    #     self.pf_stable_ok = msg.data

    def scan_cb(self, msg: LaserScan):
        ts = time.time()

        # ---------------------- 추가 부분(충돌 처리 FSM) ----------------------
        if self.collision:

            elapsed = time.time() - self.collision_start_time

            # 1) 정지 단계
            if self.brake_phase:
                if elapsed < self.brake_hold_duration:
                    out = AckermannDriveStamped()
                    out.header.stamp = msg.header.stamp
                    out.drive.steering_angle = 0.0
                    out.drive.speed = 0.0

                    self.pub.publish(out)

                    # 정지 상태에선 딥러닝 무시
                    return

                # 정지 끝 → 후진 단계 진입
                self.brake_phase = False
                self.reverse_phase = True
                self.collision_start_time = time.time()  # 시간 리셋
                self.steer_ema = 0.0   
                self.get_logger().info("⏹ 정지 완료 → 후진 단계 진입")
                return

            # 2) 후진 단계 (조향 감쇠 버전)
            if self.reverse_phase:
                rev_elapsed = time.time() - self.collision_start_time

                if rev_elapsed < self.reverse_duration:

                    # ----------- 라이다 읽기 -----------
                    rng = np.asarray(msg.ranges, dtype=np.float64)
                    rng = np.nan_to_num(rng, nan=5.0, posinf=5.0, neginf=0.0)
                    rng = np.clip(rng, 0.0, 2.0)   # 2m 이상은 의미 없으니 모두 2m 처리

                    # ------------ 구역 나누기 ------------
                    left  = np.min(rng[50:200])     # 왼쪽 근접
                    right = np.min(rng[480:600])    # 오른쪽 근접
                    front = np.min(rng[880:990])    # 정면 근접

                    # ----------- 최초 한 번만 조향 결정을 저장 -----------
                    if not hasattr(self, "reverse_steer_initialized"):

                        # 기본 조향 방향 선정
                        if right < left:
                            self.reverse_base_steer = +0.45
                        else:
                            self.reverse_base_steer = -0.45

                        # 정면이 너무 가까우면 더 강하게
                        if front < 0.4:
                            self.reverse_base_steer *= 1.2

                        self.reverse_steer_initialized = True
                        self.get_logger().info(f"[후진 초기 조향] steer={self.reverse_base_steer:.3f}")

                    # ----------- 감쇠 조향 -------------
                    decay_factor = np.exp(-rev_elapsed * 1.5)   # 감쇠 강도(1.5) 조절 가능
                    steer = self.reverse_base_steer * decay_factor

                    # 후진 속도
                    speed = -0.5

                    # ----------- 출력 -----------
                    out = AckermannDriveStamped()
                    out.header.stamp = msg.header.stamp
                    out.drive.steering_angle = float(steer)
                    out.drive.speed = float(speed)

                    self.pub.publish(out)

                    return

                # 후진 끝 → 원상 복귀
                self.reverse_phase = False
                self.collision = False

                #후진 직후 차량 완전 정지시키기
                stop_msg = AckermannDriveStamped()
                stop_msg.header.stamp = msg.header.stamp
                stop_msg.drive.steering_angle = 0.0
                stop_msg.drive.speed = 0.0

                self.pub.publish(stop_msg)

                # backward_done publish
                msg_done = Bool()
                msg_done.data = True
                self.backward_done_pub.publish(msg_done)
                self.get_logger().info("backward_done 발행됨")

                # 초기화 변수 삭제 (다음 충돌 대비)
                if hasattr(self, "reverse_steer_initialized"):
                    del self.reverse_steer_initialized
                if hasattr(self, "reverse_base_steer"):
                    del self.reverse_base_steer

                self.get_logger().info("🔄 후진 완료 → 딥러닝 정상 주행 복귀")
                return    

        # # NORMAL 모드면 TLN 출력 완전 정지(TLN에서 컨트롤러로 변경 시 제어가 불가능할 때 사용)
        # if ("NORMAL" in self.state) or (self.pf_stable_ok):
        #     out = AckermannDriveStamped()
        #     out.header.stamp = msg.header.stamp
        #     out.drive.steering_angle = 0.0
        #     out.drive.speed = 0.0

        #     self.pub.publish(out)
        #     self.get_logger().info("✅stop (NORMAL mode)")

        #     return

        # -------------------------------------------------------------
        self.get_logger().info("🔥tln tln tln tln tln tln tln tln tln tln")
        rng = np.asarray(msg.ranges, dtype=np.float64)
        rng = self.make_hokuyo_scan(rng)

        # 라이다 클램핑: NaN/±Inf 처리 후 [0, clamp_range_max_m]로 제한
        cap = float(self.clamp_range_max_m)
        if cap <= 0.0:
            cap = 10.0  # 방어적 기본값
        rng = np.nan_to_num(rng, nan=cap, posinf=cap, neginf=0.0)
        rng = np.clip(rng, 0.0, cap)

        # 역방향이면 입력 LiDAR 좌우 반전
        if self.reverse_direction:
            rng = rng[::-1].copy()

        arr = rng[::self.down_sample].reshape(1, -1, 1)
        steering_raw, speed_raw = self.dnn_output(arr)

        # 역방향이면 출력 조향만 부호 반전
        if self.reverse_direction:
            steering_raw = -steering_raw

        # ====== 보정(EMA/데드존/커브시 속도 감쇠/클립) ======
        steering = self.steer_ema = (1.0 - self.ema_alpha) * self.steer_ema + self.ema_alpha * steering_raw
        if abs(steering) < self.steering_deadzone_rad:
            steering = 0.0

        model_speed = self.linear_map(speed_raw, -1, 1, self.v_min, self.v_max)
        turn_factor = min(abs(steering) / self.steer_abs_max, 1.0)
        curve_damping_ratio = 1.0 - turn_factor
        final_speed = model_speed * curve_damping_ratio
        speed = (1.0 - self.curve_speed_weight) * model_speed + self.curve_speed_weight * final_speed

        steering = float(np.clip(steering, -self.steer_abs_max, self.steer_abs_max))
        speed    = float(np.clip(speed,    self.v_min,          self.v_max))

        # 매핑 모드일 때 속도만 1.0~3.0 m/s로 제한
        if self.mapping:
            if speed < 1.0: speed = 1.0
            if speed > 3.0: speed = 3.0
        # ===================================================

        out = AckermannDriveStamped()
        out.header.stamp = msg.header.stamp
        out.drive.steering_angle = steering
        out.drive.speed = speed

        if self.drive_publish_enabled:
            self.pub.publish(out)
            print(f"Servo: {steering:.4f}, Speed: {speed:.3f} m/s | Took: {(time.time() - ts) * 1000:.2f} ms")

    # ------------------ 추가 부분 ------------------
    def collision_cb(self, msg):

        # 새로운 충돌 시작일 때만 처리
        if msg.data and not self.collision:
            self.collision = True
            self.brake_phase = True
            self.reverse_phase = False
            self.collision_start_time = time.time()
            self.get_logger().warn("⚠ 충돌 감지 → 후진 모드 진입")

        # msg.data = False는 의미 없음 (복귀는 scan_cb에서 처리)
    # ------------------------------------------------


    # 파라미터 변경 콜백
    def _on_params(self, params):
        for p in params:
            if p.name == "drive_publish_enabled" and p.type_ == Parameter.Type.BOOL:
                self.drive_publish_enabled = bool(p.value)
            elif p.name == 'scan_topic':
                self.scan_topic = str(p.value)  # 런타임 교체는 재시작 권장
            elif p.name == 'drive_topic':
                self.drive_topic = str(p.value)
            elif p.name == 'down_sample':
                self.down_sample = int(p.value)
            elif p.name == 'ema_alpha':
                self.ema_alpha = float(p.value)
            elif p.name == 'steering_deadzone_rad':
                self.steering_deadzone_rad = float(p.value)
            elif p.name == 'reverse_direction':
                self.reverse_direction = bool(p.value)
            elif p.name == 'curve_speed_weight':
                self.curve_speed_weight = float(p.value)
            elif p.name == 'v_min':
                self.v_min = float(p.value)
            elif p.name == 'v_max':
                self.v_max = float(p.value)
            elif p.name == 'steer_abs_max':
                self.steer_abs_max = float(p.value)
            elif p.name == 'mapping':
                self.mapping = bool(p.value)
                self.get_logger().info(f"[tln_inference] mapping = {self.mapping}")
            elif p.name == 'clamp_range_max_m':
                self.clamp_range_max_m = float(p.value)
                self.get_logger().info(f"[tln_inference] clamp_range_max_m = {self.clamp_range_max_m:.2f} m")
        return SetParametersResult(successful=True)

def main(args=None):
    rclpy.init(args=args)
    node = TLNInference()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
