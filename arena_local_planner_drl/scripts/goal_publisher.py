from urllib import parse
import rospy
import argparse

from geometry_msgs.msg import PoseStamped


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--x", "-x", type=int)
    parser.add_argument("--y", "-y", type=int)
    parser.add_argument("--z", "-z", type=int)

    return parser.parse_known_args()


def main():
    rospy.init_node("goal_publisher_node")

    args, _ = parse_arguments()

    publisher = rospy.Publisher("/global_goal", PoseStamped, queue_size=1)

    msg = PoseStamped()
    msg.pose.position.x = args.x
    msg.pose.position.y = args.y
    msg.pose.position.z = args.z

    rate = rospy.Rate(5)

    while not rospy.is_shutdown():
        publisher.publish(msg)

        rate.sleep()


if __name__ == "__main__":
    main()