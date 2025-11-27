// Copyright (c) TensorStack, Advanced Micro Devices. All rights reserved.
// Licensed under the Apache 2.0 License.
using Microsoft.ML.OnnxRuntime;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using TensorStack.Common;

namespace TensorStack.Providers
{
    /// <summary>
    /// RyzenAI NPU provider with DirectML GPU fallback
    /// </summary>
    public static class Provider
    {
        private static bool _isInitialized;
        private const string _providerName = "RyzenAIExecutionProvider";

        /// <summary>
        /// Initializes the Provider 
        /// </summary>
        public static void Initialize()
        {
            if (_isInitialized)
                return;

            _isInitialized = true;
            DeviceManager.Initialize("DMLExecutionProvider");
        }


        /// <summary>
        /// Initializes the Provider with the specified environment options.
        /// </summary>
        /// <param name="environmentOptions">The environment options.</param>
        public static void Initialize(EnvironmentCreationOptions environmentOptions)
        {
            if (_isInitialized)
                return;

            _isInitialized = true;
            DeviceManager.Initialize(environmentOptions, "DMLExecutionProvider");
        }


        /// <summary>
        /// Gets the name of the provider.
        /// </summary>
        public static string ProviderName => _providerName;


        /// <summary>
        /// Gets the devices.
        /// </summary>
        public static IReadOnlyList<Device> GetDevices()
        {
            Initialize(); // Ensure Initialized
            return DeviceManager.Devices;
        }


        /// <summary>
        /// Gets the best device.
        /// </summary>
        /// <param name="deviceType">Type of the device.</param>
        public static Device GetDevice()
        {
            return GetDevice(DeviceType.NPU);
        }


        /// <summary>
        /// Gets the best device.
        /// </summary>
        /// <param name="deviceType">Type of the device.</param>
        public static Device GetDevice(DeviceType deviceType)
        {
            if (deviceType == DeviceType.NPU)
                return GetDevices().FirstOrDefault(x => x.Type == DeviceType.GPU);

            return GetDevices().FirstOrDefault(x => x.Type == deviceType);
        }


        /// <summary>
        /// Gets the Device by DeviceId.
        /// </summary>
        /// <param name="deviceType">Type of the device.</param>
        /// <param name="deviceId">The device identifier.</param>
        public static Device GetDevice(DeviceType deviceType, int deviceId)
        {
            if (deviceType == DeviceType.NPU)
                return GetDevices().FirstOrDefault(x => x.Type == DeviceType.GPU && x.DeviceId == deviceId);

            return GetDevices().FirstOrDefault(x => x.Type == deviceType && x.DeviceId == deviceId);
        }


        /// <summary>
        /// Gets the RyzenAI provider this DeviceType.
        /// </summary>
        /// <param name="optimizationLevel">The optimization level.</param>
        public static ExecutionProvider GetProvider(GraphOptimizationLevel optimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL)
        {
            return GetDevice().GetProvider(optimizationLevel);
        }


        /// <summary>
        /// Gets the RyzenAI provider this DeviceType.
        /// </summary>
        /// <param name="deviceType">Type of the device.</param>
        /// <param name="optimizationLevel">The optimization level.</param>
        public static ExecutionProvider GetProvider(DeviceType deviceType, GraphOptimizationLevel optimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL)
        {
            return GetDevice(deviceType).GetProvider(optimizationLevel);
        }


        /// <summary>
        /// Gets the RyzenAI provider for NPU if supported, else DirectML GPU fallback.
        /// </summary>
        /// <param name="deviceType">Type of the device.</param>
        /// <param name="deviceId">The device identifier.</param>
        /// <param name="optimizationLevel">The optimization level.</param>
        public static ExecutionProvider GetProvider(DeviceType deviceType, int fallbackDeviceId, GraphOptimizationLevel optimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL)
        {
            return GetDevice(deviceType, fallbackDeviceId).GetProvider(optimizationLevel);
        }


        /// <summary>
        /// Gets the RyzenAI provider for this Device.
        /// </summary>
        /// <param name="device">The device.</param>
        /// <param name="optimizationLevel">The optimization level.</param>
        public static ExecutionProvider GetProvider(this Device device, GraphOptimizationLevel optimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL)
        {
            if (device == null)
                return default;
            else if (device.Type == DeviceType.CPU)
                return CreateCPUProvider(optimizationLevel);

            return CreateRyzenProvider(device.DeviceId, optimizationLevel);
        }


        /// <summary>
        /// Gets the RyzenAI provider for NPU if supported, else DirectML fallback.
        /// </summary>
        /// <param name="fallbackDeviceId">The fallback device identifier.</param>
        /// <param name="optimizationLevel">The optimization level.</param>
        private static ExecutionProvider CreateRyzenProvider(int fallbackDeviceId, GraphOptimizationLevel optimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL)
        {
            var memoryInfo = new OrtMemoryInfo(OrtMemoryInfo.allocatorCPU, OrtAllocatorType.DeviceAllocator, fallbackDeviceId, OrtMemType.Default);
            return new ExecutionProvider(_providerName, memoryInfo, configuration =>
            {
                var sessionOptions = new SessionOptions
                {
                    GraphOptimizationLevel = optimizationLevel
                };

                var modelCache = Path.Combine(Path.GetDirectoryName(configuration.Path), ".cache");
                if (Directory.Exists(modelCache))
                    sessionOptions.AddSessionConfigEntry("dd_cache", modelCache);

                sessionOptions.AddSessionConfigEntries(configuration.SessionOptions);
                sessionOptions.RegisterCustomOpLibrary("onnx_custom_ops.dll");
                sessionOptions.AppendExecutionProvider_DML();
                sessionOptions.AppendExecutionProvider_CPU();
                return sessionOptions;
            });
        }


        /// <summary>
        /// Gets the CPU provider.
        /// </summary>
        /// <param name="optimizationLevel">The optimization level.</param>
        /// <returns>ExecutionProvider.</returns>
        private static ExecutionProvider CreateCPUProvider(GraphOptimizationLevel optimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL)
        {
            return new ExecutionProvider(DeviceManager.CPUProviderName, OrtMemoryInfo.DefaultInstance, configuration =>
            {
                var sessionOptions = new SessionOptions
                {
                    EnableCpuMemArena = true,
                    EnableMemoryPattern = true,
                    GraphOptimizationLevel = optimizationLevel
                };

                sessionOptions.AddSessionConfigEntries(configuration.SessionOptions);
                sessionOptions.AppendExecutionProvider_CPU();
                return sessionOptions;
            });
        }
    }

}
