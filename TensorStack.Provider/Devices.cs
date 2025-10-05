using Microsoft.ML.OnnxRuntime;
using System;
using System.Collections.Generic;
using System.Linq;

namespace TensorStack.Providers
{
    public static class Devices
    {
        private static OrtEnv _environment;
        private static EnvironmentCreationOptions _environmentOptions;

        /// <summary>
        /// Initializes this instance.
        /// </summary>
        public static void Initialize()
        {
            Initialize(new EnvironmentCreationOptions
            {
                logId = "TensorStack",
                threadOptions = new OrtThreadingOptions
                {
                    GlobalSpinControl = true,
                    GlobalInterOpNumThreads = 1,
                    GlobalIntraOpNumThreads = 1
                }
            });
        }


        /// <summary>
        /// Initializes the specified environment options.
        /// </summary>
        /// <param name="environmentOptions">The environment options.</param>
        public static void Initialize(EnvironmentCreationOptions environmentOptions)
        {
            _environmentOptions = environmentOptions;
            _environment = OrtEnv.CreateInstanceWithOptions(ref _environmentOptions);
        }


        /// <summary>
        /// Gets the devices.
        /// </summary>
        /// <param name="executionProvider">The execution provider.</param>
        /// <param name="libraryPath">The library path.</param>
        public static IReadOnlyList<Device> GetDevices(string executionProvider, string libraryPath = default)
        {
            if (_environment == null)
                Initialize();

            var providers = _environment.GetAvailableProviders();
            if (!providers.Contains(executionProvider, StringComparer.OrdinalIgnoreCase))
                return [];

            if (!string.IsNullOrEmpty(executionProvider))
                _environment.RegisterExecutionProviderLibrary(executionProvider, libraryPath);

            var devices = new List<Device>();
            foreach (var epDevice in _environment.GetEpDevices())
            {
                if (!epDevice.EpName.Equals(executionProvider, StringComparison.OrdinalIgnoreCase))
                    continue;

                devices.Add(CreateDevice(epDevice));
            }
            return devices;
        }


        /// <summary>
        /// Creates the device.
        /// </summary>
        /// <param name="epDevice">The ep device.</param>
        /// <returns>Device.</returns>
        private static Device CreateDevice(OrtEpDevice epDevice)
        {
            var device = epDevice.HardwareDevice;
            var metadata = device.Metadata.Entries;
            return new Device
            {
                Type = Enum.Parse<DeviceType>(device.Type.ToString()),
                Name = metadata.ParseOrDefault("Description", string.Empty),
                Memory = metadata.ParseOrDefault("DxgiVideoMemory", 0, " MB"),
                AdapterIndex = metadata.ParseOrDefault("DxgiAdapterNumber", 0),
                PerformanceIndex = metadata.ParseOrDefault("DxgiHighPerformanceIndex", 0),
                HardwareLUID = metadata.ParseOrDefault("LUID", 0),
                HardwareID = (int)device.DeviceId,
                HardwareVendor = device.Vendor,
                HardwareVendorId = (int)device.VendorId,
            };
        }


        /// <summary>
        /// Parse Metadata values
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="metadata">The metadata.</param>
        /// <param name="key">The key.</param>
        /// <param name="defaultValue">The default value.</param>
        /// <param name="replace">The replace.</param>
        /// <returns>T.</returns>
        private static T ParseOrDefault<T>(this IReadOnlyDictionary<string, string> metadata, string key, T defaultValue, string replace = null)
        {
            if (!metadata.ContainsKey(key))
                return defaultValue;

            var value = metadata[key].Trim();
            if (!string.IsNullOrEmpty(replace))
                value = value.Replace(replace, string.Empty);

            if (typeof(T) == typeof(string))
            {
                return (T)(object)value;
            }
            else if (typeof(T) == typeof(int))
            {
                if (!int.TryParse(value, out var intResult))
                    return defaultValue;

                return (T)(object)intResult;
            }
            else if (typeof(T) == typeof(Enum))
            {
                if (!Enum.TryParse(typeof(T), value, out var enumResult))
                    return defaultValue;

                return (T)enumResult;
            }
            return defaultValue;
        }
    }
}
