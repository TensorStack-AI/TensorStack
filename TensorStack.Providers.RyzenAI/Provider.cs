// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using Microsoft.ML.OnnxRuntime;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using TensorStack.Common;

namespace TensorStack.Providers
{
    public static class Provider
    {
        private static bool _isInitialized;
        private const string _providerName = "DMLExecutionProvider";


        /// <summary>
        /// Initializes the Provider 
        /// </summary>
        public static void Initialize()
        {
            if (_isInitialized)
                return;

            _isInitialized = true;
            DeviceManager.Initialize(_providerName);
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
            DeviceManager.Initialize(environmentOptions, _providerName);
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
            return GetDevices().FirstOrDefault(x => x.Type == deviceType);
        }


        /// <summary>
        /// Gets the Device by DeviceId.
        /// </summary>
        /// <param name="deviceType">Type of the device.</param>
        /// <param name="deviceId">The device identifier.</param>
        public static Device GetDevice(DeviceType deviceType, int deviceId)
        {
            return GetDevices().FirstOrDefault(x => x.Type == deviceType && x.DeviceId == deviceId);
        }


        /// <summary>
        /// Gets the DirectML provider this DeviceType.
        /// </summary>
        /// <param name="optimizationLevel">The optimization level.</param>
        public static ExecutionProvider GetProvider(GraphOptimizationLevel optimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL)
        {
            return GetDevice().GetProvider(optimizationLevel);
        }


        /// <summary>
        /// Gets the DirectML provider this DeviceType.
        /// </summary>
        /// <param name="deviceType">Type of the device.</param>
        /// <param name="optimizationLevel">The optimization level.</param>
        public static ExecutionProvider GetProvider(DeviceType deviceType, GraphOptimizationLevel optimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL)
        {
            return GetDevice(deviceType).GetProvider(optimizationLevel);
        }


        /// <summary>
        /// Gets the DirectML provider this DeviceType, DeviceId.
        /// </summary>
        /// <param name="deviceType">Type of the device.</param>
        /// <param name="deviceId">The device identifier.</param>
        /// <param name="optimizationLevel">The optimization level.</param>
        public static ExecutionProvider GetProvider(DeviceType deviceType, int deviceId, GraphOptimizationLevel optimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL)
        {
            return GetDevice(deviceType, deviceId).GetProvider(optimizationLevel);
        }


        /// <summary>
        /// Gets the DirectML provider for this Device.
        /// </summary>
        /// <param name="device">The device.</param>
        /// <param name="optimizationLevel">The optimization level.</param>
        public static ExecutionProvider GetProvider(this Device device, GraphOptimizationLevel optimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL)
        {
            if (device == null)
                return default;
            else if (device.Type == DeviceType.NPU)
                return CreateRyzenProvider(device.DeviceId, optimizationLevel);
            else if (device.Type == DeviceType.CPU)
                return CreateCPUProvider(optimizationLevel);

            return CreateDMLProvider(device.DeviceId, optimizationLevel);
        }


        /// <summary>
        /// Gets the DirectML provider for this DeviceId.
        /// </summary>
        /// <param name="deviceId">The device identifier.</param>
        /// <param name="optimizationLevel">The optimization level.</param>
        private static ExecutionProvider CreateRyzenProvider(int deviceId, GraphOptimizationLevel optimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL)
        {
            var memoryInfo = new OrtMemoryInfo(OrtMemoryInfo.allocatorCPU, OrtAllocatorType.DeviceAllocator, deviceId, OrtMemType.Default);
            return new ExecutionProvider(_providerName, memoryInfo, configuration =>
            {
                var sessionOptions = new SessionOptions
                {
                    GraphOptimizationLevel = optimizationLevel
                };

                var modelCache = Path.Combine(Path.GetDirectoryName(configuration.Path), ".cache");
                if (Directory.Exists(modelCache))
                    sessionOptions.AddSessionConfigEntry("dd_cache", modelCache);

                if (!configuration.SessionOptions.IsNullOrEmpty())
                {
                    foreach (var sessionOption in configuration.SessionOptions)
                    {
                        sessionOptions.AddSessionConfigEntry(sessionOption.Key, sessionOption.Value);
                    }
                }

                sessionOptions.RegisterCustomOpLibrary("onnx_custom_ops.dll");
                sessionOptions.AppendExecutionProvider_CPU();
                sessionOptions.AppendExecutionProvider_CPU();
                return sessionOptions;
            });
        }


        /// <summary>
        /// Gets the DirectML provider for this DeviceId.
        /// </summary>
        /// <param name="deviceId">The device identifier.</param>
        /// <param name="optimizationLevel">The optimization level.</param>
        private static ExecutionProvider CreateDMLProvider(int deviceId, GraphOptimizationLevel optimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL)
        {
            var memoryInfo = new OrtMemoryInfo(OrtMemoryInfo.allocatorCPU, OrtAllocatorType.DeviceAllocator, deviceId, OrtMemType.Default);
            return new ExecutionProvider(_providerName, memoryInfo, configuration =>
            {
                var sessionOptions = new SessionOptions
                {
                    GraphOptimizationLevel = optimizationLevel
                };

                sessionOptions.AppendExecutionProvider_DML(deviceId);
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
                sessionOptions.AppendExecutionProvider_CPU();
                return sessionOptions;
            });
        }
    }

}
