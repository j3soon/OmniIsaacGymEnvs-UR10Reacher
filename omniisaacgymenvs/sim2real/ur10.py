import asyncio
import math

import numpy as np

try:
    import rospy
    # Ref: https://github.com/ros-controls/ros_controllers/blob/melodic-devel/rqt_joint_trajectory_controller/src/rqt_joint_trajectory_controller/joint_trajectory_controller.py
    from control_msgs.msg import JointTrajectoryControllerState
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
except ImportError:
    rospy = None

class RealWorldUR10():
    # Defined in ur10.usd
    sim_dof_angle_limits = [
        (-360, 360, False),
        (-360, 360, False),
        (-360, 360, False),
        (-360, 360, False),
        (-360, 360, False),
        (-360, 360, False),
    ] # _sim_dof_limits[:,2] == True indicates inversed joint angle compared to real

    # Ref: https://github.com/ros-industrial/universal_robot/issues/112
    pi = math.pi
    servo_angle_limits = [
        (-2*pi, 2*pi),
        (-2*pi, 2*pi),
        (-2*pi, 2*pi),
        (-2*pi, 2*pi),
        (-2*pi, 2*pi),
        (-2*pi, 2*pi),
    ]
    # ROS-related strings
    state_topic = '/scaled_pos_joint_traj_controller/state'
    cmd_topic = '/scaled_pos_joint_traj_controller/command'
    joint_names = [
        'elbow_joint',
        'shoulder_lift_joint',
        'shoulder_pan_joint',
        'wrist_1_joint',
        'wrist_2_joint',
        'wrist_3_joint'
    ]
    # Joint name mapping to simulation action index
    joint_name_to_idx = {
        'elbow_joint': 2,
        'shoulder_lift_joint': 1,
        'shoulder_pan_joint': 0,
        'wrist_1_joint': 3,
        'wrist_2_joint': 4,
        'wrist_3_joint': 5
    }
    def __init__(self, fail_quietely=False, verbose=False) -> None:
        print("Connecting to real-world UR10")
        self.fail_quietely = fail_quietely
        self.verbose = verbose
        self.pub_freq = 10 # Hz
        # Not really sure if current_pos and target_pos require mutex here.
        self.current_pos = None
        self.target_pos = None
        if rospy is None:
            if not self.fail_quietely:
                raise ValueError("ROS is not installed!")
            print("ROS is not installed!")
            return
        try:
            rospy.init_node("custom_controller", anonymous=True, disable_signals=True, log_level=rospy.ERROR)
        except rospy.exceptions.ROSException as e:
            print("Node has already been initialized, do nothing")
        if self.verbose:
            print("Receiving real-world UR10 joint angles...")
            print("If you didn't see any outputs, you may have set up UR5 or ROS incorrectly.")
        self.sub = rospy.Subscriber(
            self.state_topic,
            JointTrajectoryControllerState,
            self.sub_callback,
            queue_size=1
        )
        self.pub = rospy.Publisher(
            self.cmd_topic,
            JointTrajectory,
            queue_size=1
        )
        # self.min_traj_dur = 5.0 / self.pub_freq  # Minimum trajectory duration
        self.min_traj_dur = 0  # Minimum trajectory duration

        # For catching exceptions in asyncio
        def custom_exception_handler(loop, context):
            print(context)
        # Ref: https://docs.python.org/3/library/asyncio-eventloop.html#asyncio.loop.set_exception_handler
        asyncio.get_event_loop().set_exception_handler(custom_exception_handler)
        # Ref: https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/tutorial_ros_custom_message.html
        asyncio.ensure_future(self.pub_task())
    
    def sub_callback(self, msg):
        # msg has type: JointTrajectoryControllerState
        actual_pos = {}
        for i in range(len(msg.joint_names)):
            joint_name = msg.joint_names[i]
            joint_pos = msg.actual.positions[i]
            actual_pos[joint_name] = joint_pos
        self.current_pos = actual_pos
        if self.verbose:
            print(f'(sub) {actual_pos}')
    
    async def pub_task(self):
        while not rospy.is_shutdown():
            await asyncio.sleep(1.0 / self.pub_freq)
            if self.current_pos is None:
                # Not ready (recieved UR state) yet
                continue
            if self.target_pos is None:
                # No command yet
                continue
            # Construct message
            dur = [] # move duration of each joints
            traj = JointTrajectory()
            traj.joint_names = self.joint_names
            point = JointTrajectoryPoint()
            moving_average = 1
            for name in traj.joint_names:
                pos = self.current_pos[name]
                cmd = pos * (1-moving_average) + self.target_pos[self.joint_name_to_idx[name]] * moving_average
                max_vel = 3.15 # from ur5.urdf (or ur5.urdf.xacro)
                duration = abs(cmd - pos) / max_vel # time = distance / velocity
                dur.append(max(duration, self.min_traj_dur))
                point.positions.append(cmd)
            point.time_from_start = rospy.Duration(max(dur))
            traj.points.append(point)
            self.pub.publish(traj)
            print(f'(pub) {point.positions}')

    def send_joint_pos(self, joint_pos):
        if len(joint_pos) != 6:
            raise Exception("The length of UR10 joint_pos is {}, but should be 6!".format(len(joint_pos)))

        # Convert Sim angles to Real angles
        target_pos = [0] * 6
        for i, pos in enumerate(joint_pos):
            if i == 5:
                # Ignore the gripper joints for Reacher task
                continue
            # Map [L, U] to [A, B]
            L, U, inversed = self.sim_dof_angle_limits[i]
            A, B = self.servo_angle_limits[i]
            angle = np.rad2deg(float(pos))
            if not L <= angle <= U:
                print("The {}-th simulation joint angle ({}) is out of range! Should be in [{}, {}]".format(i, angle, L, U))
                angle = np.clip(angle, L, U)
            target_pos[i] = (angle - L) * ((B-A)/(U-L)) + A # Map [L, U] to [A, B]
            if inversed:
                target_pos[i] = (B-A) - (target_pos[i] - A) + A # Map [A, B] to [B, A]
            if not A <= target_pos[i] <= B:
                raise Exception("(Should Not Happen) The {}-th real world joint angle ({}) is out of range! hould be in [{}, {}]".format(i, target_pos[i], A, B))
            self.target_pos = target_pos

if __name__ == "__main__":
    print("Make sure you are running `roslaunch ur_robot_driver`.")
    print("If the machine running Isaac is not the ROS master node, " + \
          "make sure you have set the environment variables: " + \
          "`ROS_MASTER_URI` and `ROS_HOSTNAME`/`ROS_IP` correctly.")
    ur10 = RealWorldUR10(verbose=True)
    rospy.spin()
