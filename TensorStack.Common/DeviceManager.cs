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
        private readonly static byte[] _validationModel = [8, 10, 18, 0, 58, 73, 10, 18, 10, 1, 120, 10, 1, 107, 18, 1, 118, 18, 1, 105, 34, 4, 84, 111, 112, 75, 18, 1, 116, 90, 9, 10, 1, 120, 18, 4, 10, 2, 8, 1, 90, 15, 10, 1, 107, 18, 10, 10, 8, 8, 7, 18, 4, 10, 2, 8, 1, 98, 9, 10, 1, 118, 18, 4, 10, 2, 8, 1, 98, 9, 10, 1, 105, 18, 4, 10, 2, 8, 7, 66, 2, 16, 21];
        private static OrtEnv _environment;
        private static EnvironmentCreationOptions _environmentOptions;
        private static IReadOnlyList<Device> _devices;
        private static string _deviceProvider;

        /// <summary>
        /// Initializes this instance.
        /// </summary>
        public static void Initialize(string executionProvider, Func<Device, SessionOptions> sessionValidator, string libraryPath = default)
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
            }, executionProvider, sessionValidator, libraryPath);
        }


        /// <summary>
        /// Initializes the specified environment options.
        /// </summary>
        /// <param name="environmentOptions">The environment options.</param>
        public static void Initialize(EnvironmentCreationOptions environmentOptions, string executionProvider, Func<Device, SessionOptions> sessionValidator, string libraryPath = default)
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
            _devices = ValidateDevices(devices, sessionValidator);
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
                Vendor = Enum.IsDefined(typeof(VendorType), (int)device.VendorId)
                    ? (VendorType)(int)device.VendorId
                    : VendorType.CPU
            };
        }


        /// <summary>
        /// Validates the devices.
        /// </summary>
        /// <param name="devices">The devices.</param>
        /// <param name="sessionValidator">The session validator.</param>
        private static IReadOnlyList<Device> ValidateDevices(List<Device> devices, Func<Device, SessionOptions> sessionValidator)
        {
            if (sessionValidator == null)
                return devices;

            var validDevices = new List<Device>();
            foreach (var device in devices)
            {
                try
                {
                    var sessionOptions = sessionValidator(device);
                    if (sessionOptions == null)
                        continue;

                    using (sessionOptions)
                    using (var inferenceSession = new InferenceSession(_validationModel, sessionOptions))
                    {
                        validDevices.Add(device);
                    }
                }
                catch (Exception) { }
            }
            return validDevices;
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
