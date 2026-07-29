import time
import mujoco
import mujoco.viewer
from threading import Thread
import threading

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py_bridge import UnitreeSdk2Bridge, ElasticBand

import config


locker = threading.Lock()

mj_model = mujoco.MjModel.from_xml_path(config.ROBOT_SCENE)
mj_data = mujoco.MjData(mj_model)


def FindBodyId(model, body_names):
    for body_name in body_names:
        try:
            return model.body(body_name).id
        except KeyError:
            continue
    raise KeyError(f"None of these bodies exist in the model: {body_names}")


if config.ENABLE_ELASTIC_BAND:
    elastic_band = ElasticBand()
    band_attached_link = FindBodyId(
        mj_model, ["torso_link", "base_link", "pelvis_link", "pelvis"]
    )
    viewer = mujoco.viewer.launch_passive(
        mj_model, mj_data, key_callback=elastic_band.MujuocoKeyCallback
    )
else:
    viewer = mujoco.viewer.launch_passive(mj_model, mj_data)

mj_model.opt.timestep = config.SIMULATE_DT
num_motor_ = mj_model.nu
dim_motor_sensor_ = 3 * num_motor_

time.sleep(0.2)


def SimulationThread():
    global mj_data, mj_model

    ChannelFactoryInitialize(config.DOMAIN_ID, config.INTERFACE)
    unitree = UnitreeSdk2Bridge(mj_model, mj_data)

    if config.USE_JOYSTICK:
        unitree.SetupJoystick(device_id=0, js_type=config.JOYSTICK_TYPE)
    if config.PRINT_SCENE_INFORMATION:
        unitree.PrintSceneInformation()

    while viewer.is_running():
        step_start = time.perf_counter()

        locker.acquire()

        if config.ENABLE_ELASTIC_BAND:
            mj_data.xfrc_applied[band_attached_link, :] = 0
            if elastic_band.enable:
                force, torque = elastic_band.Advance(
                    mj_model, mj_data, band_attached_link
                )
                mj_data.xfrc_applied[band_attached_link, :3] = force
                mj_data.xfrc_applied[band_attached_link, 3:] = torque
        mujoco.mj_step(mj_model, mj_data)

        locker.release()

        time_until_next_step = mj_model.opt.timestep - (
            time.perf_counter() - step_start
        )
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)


def PhysicsViewerThread():
    while viewer.is_running():
        locker.acquire()
        viewer.sync()
        locker.release()
        time.sleep(config.VIEWER_DT)


if __name__ == "__main__":
    viewer_thread = Thread(target=PhysicsViewerThread)
    sim_thread = Thread(target=SimulationThread)

    viewer_thread.start()
    sim_thread.start()
