"""
Collection of environment classes that are based on rai-python
"""
import time
import numpy as np

# rai-python: https://github.com/MarcToussaint/rai-python
import libry as ry


class BoxEnv():
    """
    Robotic environment similar to the 'FetchPush-v1'
    environment in the open ai gym baseline
    """

    def __init__(
            self,
            action_duration=0.5,
            floor_level=0.65,
            finger_relative_level=0.14,
            tau=.01,
            file=None,
            display=False
    ):
        self.action_duration = action_duration
        self.floor_level = floor_level
        self.finger_relative_level = finger_relative_level
        self.tau = tau

        self.n_steps = int(self.action_duration/self.tau)
        self.proportion_per_step = 1/self.n_steps

        self.target_tolerance = 0.1

        self.config = ry.Config()

        if file is not None:
            self.config.addFile(file)
        else:
            self.config.addFile('z.push_box.g')

        self.config.makeObjectsFree(['finger'])
        self.config.setJointState([0.3, 0.3, 0.15, 1., 0., 0., 0.])

        self.finger_radius = self.config.frame('finger').info()['size'][0]

        self.simulation = self.config.simulation(
            ry.SimulatorEngine.physx, display)

        self.reset_box()
        self.box_dimensions = [0.4, 0.4, 0.2, 0.05]
        self.reset([0.3, 0.3])

        self.maximum_xy_for_finger = 1.7
        self.minimum_rel_z_for_finger = 0.05 + 0.03
        self.maximum_rel_z_for_finger = 1

        self.config.frame('floor').setColor(
            np.array((200, 200, 200))/255,
        )

        rgb = [93, 87, 94]
        self.config.frame('finger').setColor(np.array([
            *rgb, 255
        ])/255)
        self.config.frame('box').setColor(np.array([
            *rgb, 255
        ])/255)

    def view(self):
        """
        Create interactive view of current configuration
        """
        return self.config.view()

    def add_and_show_target(self, target_state):
        """
        Add target state and visualize it in view
        """
        self.config.delFrame('target')
        target = self.config.addFrame(name="target")
        target.setShape(ry.ST.sphere, size=[self.target_tolerance])
        target.setColor([1, 1, 0, 0.4])

        self.set_frame_state(
            target_state,
            "target"
        )
        self.config.frame('target').setColor(
            np.array((81, 203, 32, 130))/255
        )

    def reset_box(self, coords=(0, 0)):
        """
        Reset the box to an arbitrary position
        """
        self.set_frame_state(
            coords,
            'box'
        )
        state_now = self.config.getFrameState()
        self.simulation.setState(state_now, np.zeros((state_now.shape[0], 6)))

    def sample_box_position(self):
        """
        Sample random position for the box on the table
        """
        return 2.6*np.random.rand(2) - 1.3

    def reset(
            self,
            finger_position,
            box_position=None,
            goal_position=None
    ):
        """
        Reset the state (i.e. the finger state) to an arbitrary position
        """
        if box_position is None:
            box_position = self.sample_box_position()
        if goal_position is None:
            goal_position = self.sample_box_position()

        joint_q = np.array([
            *finger_position,
            self.finger_relative_level,
            1.,
            0.,
            0.,
            0.
        ])

        self.config.setJointState(joint_q)
        self.simulation.step(u_control=[0, 0, 0, 0, 0, 0, 0], tau=self.tau)
        self.reset_box(coords=box_position)
        self.add_and_show_target(goal_position)

    def evolve(
            self,
            n_steps=1000,
            fps=None
    ):
        """
        Evolve the simulation for n_steps time steps of length self.tau
        """
        for _ in range(n_steps):
            self.simulation.step(u_control=[0, 0, 0, 0, 0, 0, 0], tau=self.tau)
            if fps is not None:
                time.sleep(1/fps)

    def set_frame_state(
            self,
            state,
            frame_name
    ):
        """
        Set an arbitrary frame of the configuration to
        and arbitrary state
        """
        self.config.frame(frame_name).setPosition([
            *state[:2],
            self.floor_level
        ])
        self.config.frame(frame_name).setQuaternion(
            [1., 0., 0., 0.]
        )

    def get_state(self):
        """
        Get the current state, i.e. position of the finger as well
        as the position and Quaternion of the box
        """
        return np.concatenate([
            self.config.getJointState()[:3],
            self.config.frame('box').getPosition(),
            self.config.frame('box').getQuaternion()
        ])

    def step(
            self,
            action,
            fps=None
    ):
        """
        Simulate the system's transition under an action
        """
        # clip action
        action = np.clip(
            action,
            -0.1,
            0.1
        )

        # gradual pushing movement
        joint_q = self.config.getJointState()
        for _ in range(self.n_steps):
            new_x = joint_q[0] + self.proportion_per_step * action[0]
            if abs(new_x) < self.maximum_xy_for_finger:
                joint_q[0] = new_x

            new_y = joint_q[1] + self.proportion_per_step * action[1]
            if abs(new_y) < self.maximum_xy_for_finger:
                joint_q[1] = new_y

            new_z = joint_q[2] + self.proportion_per_step * action[2]
            if new_z < self.maximum_rel_z_for_finger and new_z > self.minimum_rel_z_for_finger:
                joint_q[2] = new_z

            self.config.setJointState(joint_q)
            self.simulation.step(u_control=[0, 0, 0, 0, 0, 0, 0], tau=self.tau)
            if fps is not None:
                time.sleep(1/fps)

        observation = {
            'observation': self.get_state(),
            'achieved_goal': self.config.frame('box').getPosition(),
            'desired_goal': self.config.frame('target').getPosition()
        }
        reward = float(np.linalg.norm(
            self.config.frame(
                'box'
            ).getPosition() - self.config.frame(
                'target'
            ).getPosition()
        ) < self.target_tolerance)
        done = False
        info = {}

        return observation, reward, done, info

    def sample_action(self):
        """
        Sample a random action
        """
        return 0.1*np.random.rand(3)-0.05


class PickAndPlaceEnv():
    """
    Pick-and-place environment with "sticky contact"
    """

    def __init__(
            self,
            action_duration=0.5,
            floor_level=0.6,
            finger_relative_level=0.14,
            contact_distance=0.116,
            sticky_radius=0.08,
            tau=.01,
            file=None,
            display=False
    ):
        self.action_duration = action_duration
        self.floor_level = floor_level
        self.finger_relative_level = finger_relative_level
        self.contact_distance = contact_distance
        self.sticky_radius = sticky_radius
        self.tau = tau

        self.n_steps = int(self.action_duration/self.tau)
        self.proportion_per_step = 1/self.n_steps
        self.target_tolerance = 0.1
        self.contact_vec = None

        self.config = ry.Config()

        if file is not None:
            self.config.addFile(file)
        else:
            self.config.addFile('z.pick_and_place.g')

        self.config.makeObjectsFree(['finger'])
        self.config.setJointState([0.3, 0.3, 0.15, 1., 0., 0., 0.])

        self.finger_radius = self.config.frame('finger').info()['size'][0]

        self.simulation = self.config.simulation(
            ry.SimulatorEngine.physx, display)

        self.reset_disk()
        self.reset([0.3, 0.3])

        self.maximum_xy_for_finger = 1.7
        self.maximum_rel_z_for_finger = 1

        self.config.frame('floor').setColor(
            np.array((200, 200, 200))/255,
        )

        rgb = [93, 87, 94]
        self.config.frame('finger').setColor(np.array([
            *rgb, 255
        ])/255)
        self.config.frame('disk').setColor(np.array([
            *rgb, 255
        ])/255)

    def get_minimum_rel_z_for_finger(self):
        """
        Get minimum z height of finger over table
        """
        minimum_rel_z_for_finger = 0.05 + 0.03
        if self.contact_vec is not None:
            minimum_rel_z_for_finger += 0.1
        return minimum_rel_z_for_finger

    def view(self):
        """
        Create view of current configuration
        """
        return self.config.view()

    def add_and_show_target(self, target_state):
        """
        Add target state and visualize it in view
        """
        self.config.delFrame('target')
        target = self.config.addFrame(name="target")
        target.setShape(ry.ST.sphere, size=[self.target_tolerance])
        target.setColor([1, 1, 0, 0.4])

        self.set_frame_state(
            target_state,
            'target'
        )
        self.config.frame('target').setColor(
            np.array((81, 203, 32, 130))/255
        )

    def reset_disk(self, coords=(0, 0)):
        """
        Reset the disk to an arbitrary position
        """
        print('Test for collision here')
        self.set_frame_state(
            coords,
            'disk'
        )
        state_now = self.config.getFrameState()
        self.simulation.setState(state_now, np.zeros((state_now.shape[0], 6)))

    def sample_disk_position(self):
        """
        Sample random position for the disk on the table
        """
        return 2.6*np.random.rand(2) - 1.3

    def reset(
            self,
            finger_position,
            disk_position=None,
            goal_position=None
    ):
        """
        Reset the state (i.e. the finger state) to an arbitrary position
        """
        self.contact_vec = None

        if disk_position is None:
            disk_position = self.sample_disk_position()
        if goal_position is None:
            goal_position = self.sample_disk_position()

        joint_q = np.array([
            *finger_position,
            self.finger_relative_level,
            1.,
            0.,
            0.,
            0.
        ])

        self.config.setJointState(joint_q)
        self.simulation.step(u_control=[0, 0, 0, 0, 0, 0, 0], tau=self.tau)
        self.reset_disk(coords=disk_position)
        self.add_and_show_target(goal_position)

    def evolve(
            self,
            n_steps=1000,
            fps=None
    ):
        """
        Evolve the simulation for n_steps time steps of length self.tau
        """
        for _ in range(n_steps):
            self.simulation.step(u_control=[0, 0, 0, 0, 0, 0, 0], tau=self.tau)
            if fps is not None:
                time.sleep(1/fps)

    def set_frame_state(
            self,
            state,
            frame_name
    ):
        """
        Set an arbitrary frame of the configuration to
        and arbitrary state
        """
        assert len(state) == 2
        self.config.frame(frame_name).setPosition([
            *state,
            self.floor_level
        ])
        self.config.frame(frame_name).setQuaternion(
            [1., 0., 0., 0.]
        )

    def get_state(self):
        """
        Get the current state, i.e. position of the finger as well
        as the position and Quaternion of the disk
        """
        return np.concatenate([
            self.config.getJointState()[:3],
            self.config.frame('disk').getPosition(),
            np.array([
                float((self.contact_vec is not None))
            ])
        ])

    def step(
            self,
            action,
            fps=None
    ):
        """
        Simulate the system's transition under an action
        """
        # clip action
        action = np.clip(
            action,
            -0.1,
            0.1
        )

        # gradual pushing movement
        joint_q = self.config.getJointState()
        for _ in range(self.n_steps):
            new_x = joint_q[0] + self.proportion_per_step * action[0]
            if abs(new_x) < self.maximum_xy_for_finger:
                joint_q[0] = new_x

            new_y = joint_q[1] + self.proportion_per_step * action[1]
            if abs(new_y) < self.maximum_xy_for_finger:
                joint_q[1] = new_y

            new_z = joint_q[2] + self.proportion_per_step * action[2]
            if (
                new_z < self.maximum_rel_z_for_finger
            ) and (
                new_z > self.get_minimum_rel_z_for_finger()
            ):
                joint_q[2] = new_z

            self.config.setJointState(joint_q)
            self.simulation.step(u_control=[0, 0, 0, 0, 0, 0, 0], tau=self.tau)
            if fps is not None:
                time.sleep(1/fps)
            if self.contact_vec is not None:
                self.config.frame('disk').setPosition(
                    self.config.frame(
                        'finger'
                    ).getPosition() - self.contact_vec
                )
                self.config.frame('disk').setQuaternion(
                    [1., 0., 0., 0.]
                )
                state_now = self.config.getFrameState()
                self.simulation.setState(
                    state_now,
                    np.zeros((state_now.shape[0], 6))
                )

        if self.contact_vec is not None:
            self.config.frame('disk').setPosition(
                self.config.frame(
                    'finger'
                ).getPosition() - self.contact_vec
            )
            self.config.frame('disk').setQuaternion(
                [1., 0., 0., 0.]
            )

        if self.contact_vec is None:

            relative = self.config.frame(
                'finger'
            ).getPosition() - self.config.frame(
                'disk'
            ).getPosition()
            inside_sticky_area = np.linalg.norm(
                relative[:2]) < self.sticky_radius
            contact_now = (
                inside_sticky_area and relative[-1] < self.contact_distance
            ) and relative[-1] > 0

            # once the contact has been established, the
            # contact vec is set
            if contact_now:
                self.contact_vec = relative.copy()

        observation = {
            'observation': self.get_state(),
            'achieved_goal': self.config.frame('disk').getPosition(),
            'desired_goal': self.config.frame('target').getPosition()
        }
        reward = float(np.linalg.norm(
            self.config.frame(
                'disk'
            ).getPosition() - self.config.frame(
                'target'
            ).getPosition()
        ) < self.target_tolerance)
        done = False
        info = {'contact_vec': self.contact_vec}

        return observation, reward, done, info

    def sample_action(self):
        """
        Sample a random action
        """
        return 0.1*np.random.rand(3)-0.05
