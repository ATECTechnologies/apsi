// After electron-builder copies Electron:
//  1. Windows: set the Apsi icon and version info on the app exe with rcedit (done here instead of
//     electron-builder's own step, which downloads a winCodeSign archive whose symlinks cannot be
//     extracted on Windows without Developer Mode).
//  2. Flip Electron fuses: no ELECTRON_RUN_AS_NODE, no NODE_OPTIONS, no --inspect, app code only from asar.
const path = require("path");
const { flipFuses, FuseVersion, FuseV1Options } = require("@electron/fuses");

exports.default = async function afterPack(ctx) {
  const plat = ctx.electronPlatformName;
  const name = ctx.packager.appInfo.productFilename;
  const exe = path.join(ctx.appOutDir, {
    win32: `${name}.exe`,
    linux: ctx.packager.executableName || name.toLowerCase(),
    darwin: `${name}.app`,
  }[plat]);
  const version = ctx.packager.appInfo.version;

  if (plat === "win32" && process.platform === "win32") {
    const rc = require("rcedit");
    const rcedit = rc.rcedit || rc.default || rc;
    await rcedit(exe, {
      icon: path.join(__dirname, "icon.ico"),
      "file-version": version,
      "product-version": version,
      "version-string": {
        ProductName: "Apsi Setup",
        FileDescription: "Apsi Setup",
        CompanyName: "Apsi",
        LegalCopyright: "Apsi authors",
        OriginalFilename: `${name}.exe`,
      },
    });
  }

  await flipFuses(exe, {
    version: FuseVersion.V1,
    resetAdHocDarwinSignature: plat === "darwin",
    [FuseV1Options.RunAsNode]: false,
    [FuseV1Options.EnableCookieEncryption]: true,
    [FuseV1Options.EnableNodeOptionsEnvironmentVariable]: false,
    [FuseV1Options.EnableNodeCliInspectArguments]: false,
    // the asar hash is embedded by electron-builder's exe-editing step on Windows, which is turned off
    [FuseV1Options.EnableEmbeddedAsarIntegrityValidation]: plat !== "win32",
    [FuseV1Options.OnlyLoadAppFromAsar]: true,
  });
};
