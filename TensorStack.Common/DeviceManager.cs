// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using Microsoft.ML.OnnxRuntime;
using System;
using System.Collections.Generic;
using System.Linq;

namespace TensorStack.Common
{
    public static class DeviceManager
    {
        private static OrtEnv _environment;
        private static EnvironmentCreationOptions _environmentOptions;
        private static IReadOnlyList<Device> _devices;
        private static string _deviceProvider;

        /// <summary>
        /// Initializes this instance.
        /// </summary>
        public static void Initialize(string executionProvider, string libraryPath = default)
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
            }, executionProvider, libraryPath);
        }


        /// <summary>
        /// Initializes the specified environment options.
        /// </summary>
        /// <param name="environmentOptions">The environment options.</param>
        public static void Initialize(EnvironmentCreationOptions environmentOptions, string executionProvider, string libraryPath = default)
        {
            if (_environment is not null)
                throw new Exception("Environment is already initialized.");

            _deviceProvider = executionProvider;
            _environmentOptions = environmentOptions;
            _environment = OrtEnv.CreateInstanceWithOptions(ref _environmentOptions);

            var providers = _environment.GetAvailableProviders();
            if (!providers.Contains(_deviceProvider, StringComparer.OrdinalIgnoreCase))
                throw new Exception($"Provider {_deviceProvider} was not found in GetAvailableProviders().");

            if (!string.IsNullOrEmpty(libraryPath))
                _environment.RegisterExecutionProviderLibrary(_deviceProvider, libraryPath);

            var devices = new List<Device>();
            foreach (var epDevice in _environment.GetEpDevices())
            {
                if (epDevice.HardwareDevice.Type == OrtHardwareDeviceType.CPU || epDevice.EpName.Equals(_deviceProvider, StringComparison.OrdinalIgnoreCase))
                    devices.Add(CreateDevice(epDevice));
            }
            _devices = devices;
        }


        /// <summary>
        /// Gets the devices.
        /// </summary>
        public static IReadOnlyList<Device> Devices => _devices;


        /// <summary>
        /// The cpu provider name
        /// </summary>
        public const string CPUProviderName = "CPUExecutionProvider";


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
                Id = metadata.ParseOrDefault("DxgiAdapterNumber", 0),
                DeviceId = metadata.ParseOrDefault("DxgiHighPerformanceIndex", 0),
                Type = Enum.Parse<DeviceType>(device.Type.ToString()),
                Name = metadata.ParseOrDefault("Description", string.Empty),
                Memory = metadata.ParseOrDefault("DxgiVideoMemory", 0, " MB"),
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
