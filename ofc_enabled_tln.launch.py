#!/usr/bin/env python3
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from pathlib import Path

def generate_launch_description():
    cfg = str(Path(get_package_share_directory('stack_master')) / 'config' / 'controller_enabled.yaml')

    return LaunchDescription([

        # TLN 단독 실행
        Node(
            package='ofc',
            executable='tln_inference',
            name='tln_inference',
            output='screen',
            parameters=[
                cfg,
                {
                    'drive_publish_enabled': True,   # TLN 즉시 주행 시작
                    'enabled': True,
                    'v_min': 0.0,
                    'v_max': 3.0
                }
            ],
        ),

        # FTG 제거
        # enabled_guard 제거
        # joy_controller 제거
        # bag_recorder 제거

    ])
