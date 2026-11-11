"""PhoXi Control adapter. Call all methods from the acquisition worker."""
import logging
from time import perf_counter

from .frames import ScanFrame

LOG = logging.getLogger(__name__)


class PhoXiScanner:
    def __init__(self):
        self._control = None
        self._device = None
        self._device_id = None
        self._previous_mode = None
        self._previous_point_cloud = None
        self._previous_texture = None
        self._was_acquiring = False

    def _get_control(self):
        from phoxi_api import PhoXiControl, PhoXiError

        if self._control is None:
            try:
                self._control = PhoXiControl()
            except (OSError, PhoXiError) as exc:
                raise RuntimeError(
                    f"Could not load the PhoXi API. Check the PhoXi Control installation: {exc}"
                ) from exc
        if not self._control.is_phoxicontrol_running():
            raise RuntimeError("Start PhoXi Control, then try again.")
        return self._control

    def discover(self):
        devices = self._get_control().get_device_list(refresh=True)
        identifiers = (entry.get("device_id") or entry.get("hw_identification")
                       or entry.get("id") for entry in devices)
        return sorted({value for value in identifiers if isinstance(value, str) and value})

    def connect(self, device_id):
        from phoxi_api import PhoXiDevice

        if self._device is not None:
            raise RuntimeError("Disconnect the current scanner first.")
        self._device = self._get_control().connect(device_id, timeout_ms=5000)
        self._device_id = device_id
        try:
            device = self._device
            self._previous_mode = device.trigger_mode()
            self._was_acquiring = device.is_acquiring()
            mode_changed = self._previous_mode != PhoXiDevice.TriggerMode.SOFTWARE
            settings = None if device.info().get("file_camera") else device.frame_settings()
            if settings is not None:
                self._previous_point_cloud = settings.PointCloud.value
                self._previous_texture = settings.Texture.value
            outputs_changed = settings is not None and (
                not self._previous_point_cloud or not self._previous_texture
            )
            if self._was_acquiring and (mode_changed or outputs_changed):
                device.stop_acquisition()
            if mode_changed:
                device.set_trigger_mode(PhoXiDevice.TriggerMode.SOFTWARE)

            if outputs_changed:
                if not self._previous_point_cloud:
                    settings.PointCloud.value = True
                if not self._previous_texture:
                    settings.Texture.value = True
            if not device.is_acquiring():
                device.start_acquisition()
        except Exception:
            try:
                self.disconnect()
            except Exception:
                LOG.exception("Could not restore scanner settings after connection failed")
            raise


    @staticmethod
    def _profile_names(records):
        """PhoXi returns profile metadata dictionaries, not just strings."""
        names = []
        for item in records:
            name = item if isinstance(item, str) else (
                item.get("name") or item.get("profile_name") or item.get("Name")
                if isinstance(item, dict) else None
            )
            if isinstance(name, str) and name and name not in names:
                names.append(name)
        return names

    def get_profiles(self):
        if self._device is None:
            raise RuntimeError("Connect a scanner first.")
        return self._profile_names(self._device.profiles()), self._device.active_profile()

    def set_profile(self, name):
        if self._device is None:
            raise RuntimeError("Connect a scanner first.")
        available, _ = self.get_profiles()
        if name not in available:
            raise ValueError(f"Profile {name!r} is not available on the connected scanner.")
        from phoxi_api import PhoXiDevice

        device = self._device
        was_acquiring = device.is_acquiring()
        if was_acquiring:
            device.stop_acquisition()
        try:
            device.set_active_profile(name)
            # Profile activation may replace acquisition settings. Maintain the
            # viewer's software-trigger contract without touching the startup profile.
            if device.trigger_mode() != PhoXiDevice.TriggerMode.SOFTWARE:
                device.set_trigger_mode(PhoXiDevice.TriggerMode.SOFTWARE)
            if not device.info().get("file_camera"):
                settings = device.frame_settings()
                if settings is not None:
                    settings.PointCloud.value = True
                    settings.Texture.value = True
        finally:
            if was_acquiring and not device.is_acquiring():
                device.start_acquisition()
        return self.get_profiles()

    def trigger(self):
        if self._device is None:
            raise RuntimeError("Connect a scanner first.")
        started = perf_counter()
        frame_id = self._device.trigger_frame(wait_accept=False, wait_grabbing_end=False)
        if frame_id < 0:
            raise RuntimeError("The scanner did not accept the software trigger. Try again.")
        frame = self._device.get_frame(frame_id, timeout_ms=15000)
        if not frame.Successful or frame.PointCloud is None:
            detail = f" Scanner messages: {frame.Messages}" if frame.Messages else ""
            raise RuntimeError(f"The triggered frame did not contain a usable point cloud.{detail}")
        return ScanFrame.from_arrays(
            frame.PointCloud, frame.Texture, source=self._device_id,
            frame_id=frame_id, elapsed_ms=(perf_counter() - started) * 1000,
        )

    def disconnect(self):
        device = self._device
        if device is None:
            return
        try:
            if self._previous_mode is not None:
                mode_changed = device.trigger_mode() != self._previous_mode
                restore_outputs = self._previous_point_cloud is False or self._previous_texture is False
                if device.is_acquiring() and (mode_changed or restore_outputs or not self._was_acquiring):
                    device.stop_acquisition()
                if mode_changed:
                    device.set_trigger_mode(self._previous_mode)
                if restore_outputs:
                    settings = device.frame_settings()
                    if settings is not None:
                        if self._previous_point_cloud is False:
                            settings.PointCloud.value = self._previous_point_cloud
                        if self._previous_texture is False:
                            settings.Texture.value = self._previous_texture
                if self._was_acquiring and not device.is_acquiring():
                    device.start_acquisition()
        finally:
            try:
                device.disconnect(logout=False, stop_acquisition=False)
            finally:
                self._device = self._device_id = self._previous_mode = None
                self._previous_point_cloud = self._previous_texture = None
                self._was_acquiring = False
