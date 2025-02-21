import habitat
from habitat.sims.habitat_simulator.actions import HabitatSimActions
import cv2


FORWARD_KEY="w"
LEFT_KEY="a"
RIGHT_KEY="d"
FINISH="f"
CONT_RAND="r"


def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]


def example():
    config=habitat.get_config("benchmark/nav/pointnav/pointnav_habitat_test.yaml")
    
    # # Modify the configuration to include BaseVelAction
    # config.defrost()
    # config.TASK.POSSIBLE_ACTIONS.append("BASE_VELOCITY_CONTROL")
    # config.TASK.ACTIONS.BASE_VELOCITY_CONTROL = habitat.config.Config()
    # config.TASK.ACTIONS.BASE_VELOCITY_CONTROL.TYPE = "BaseVelAction"
    # config.freeze()

    env = habitat.Env(config)

    print("Environment creation successful")
    observations = env.reset()
    print("Destination, distance: {:3f}, theta(radians): {:.2f}".format(
        observations["pointgoal_with_gps_compass"][0],
        observations["pointgoal_with_gps_compass"][1]))
    cv2.imshow("RGB", transform_rgb_bgr(observations["rgb"]))

    print("Agent stepping around inside environment.")

    # Define a continuous action
    # Here, 'forward_velocity' is the speed at which the agent moves forward
    # 'rotation_velocity' is the speed of rotation (angular velocity)
    action_cont = {
        "action": "BASE_VELOCITY_CONTROL",
        "forward_velocity": 0.5, # Adjust as needed
        "rotation_velocity": 30.0 # Adjust as needed, in degrees per second
    }
    # Define a continuous action, e.g., moving forward with a specific velocity
    continuous_action = {
        "action": "move_forward",
        "amount": 0.1  # Adjust this value based on the desired velocity
    }

    count_steps = 0
    while not env.episode_over:
        keystroke = cv2.waitKey(0)

        if keystroke == ord(FORWARD_KEY):
            action = HabitatSimActions.move_forward
            print("action: FORWARD")
        elif keystroke == ord(LEFT_KEY):
            action = HabitatSimActions.turn_left
            print("action: LEFT")
        elif keystroke == ord(RIGHT_KEY):
            action = HabitatSimActions.turn_right
            print("action: RIGHT")
        elif keystroke == ord(FINISH):
            action = HabitatSimActions.stop
            print("action: FINISH")
        elif keystroke == ord(CONT_RAND):
            action = continuous_action
            print("action: CONTINUE RANDOM")
        else:
            print("INVALID KEY")
            continue

        observations = env.step(action)
        count_steps += 1

        print("Destination, distance: {:3f}, theta(radians): {:.2f}".format(
            observations["pointgoal_with_gps_compass"][0],
            observations["pointgoal_with_gps_compass"][1]))
        cv2.imshow("RGB", transform_rgb_bgr(observations["rgb"]))

    print("Episode finished after {} steps.".format(count_steps))

    if (
        action == HabitatSimActions.stop
        and observations["pointgoal_with_gps_compass"][0] < 0.2
    ):
        print("you successfully navigated to destination point")
    else:
        print("your navigation was unsuccessful")


if __name__ == "__main__":
    example()