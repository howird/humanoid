{ lib
, mkShell
, vulkan-loader
, python3Packages
, wayland
}:
mkShell rec {
  name = "maniskill";

  VK_ICD_FILENAMES = "/run/opengl-driver/share/vulkan/icd.d/nvidia_icd.x86_64.json";
  __EGL_VENDOR_LIBRARY_FILENAMES = "/run/opengl-driver/share/glvnd/egl_vendor.d/10_nvidia.json";
  SAPIEN_VULKAN_LIBRARY_PATH = "/run/current-system/sw/lib/libvulkan.so.1";
  # SAPIEN_VULKAN_LIBRARY_PATH = "${vulkan-loader}/lib/libvulkan.so.1";


  LD_LIBRARY_PATH = lib.makeLibraryPath [
    "/run/opengl-driver"
  #   wayland
  ];

  venvDir = "./env";
  packages = with python3Packages; [
    venvShellHook
    python
    maniskill
    mujoco
    pybullet
    debugpy
  ] ++ (with pkgs; [
  ]);
}
