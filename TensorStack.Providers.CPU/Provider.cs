﻿// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using Microsoft.ML.OnnxRuntime;
using System.Collections.Generic;
using System.Linq;
using TensorStack.Common;

namespace TensorStack.Providers
{
    public static class Provider
    {
        private static bool _isInitialized;

        /// <summary>
        /// Initializes the Provider 
        /// </summary>
        public static void Initialize()
        {
            if (_isInitialized)
                return;

            _isInitialized = true;
            DeviceManager.Initialize(ProviderName);
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
            DeviceManager.Initialize(environmentOptions, ProviderName);
        }


        /// <summary>
        /// Gets the name of the provider.
        /// </summary>
        public static string ProviderName => DeviceManager.CPUProviderName;


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
            return GetDevice(DeviceType.CPU);
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
        /// Gets the CPU provider this DeviceType.
        /// </summary>
        /// <param name="deviceType">Type of the device.</param>
        /// <param name="optimizationLevel">The optimization level.</param>
        public static ExecutionProvider GetProvider(GraphOptimizationLevel optimizationLevel = GraphOptimizationLevel.ORT_DISABLE_ALL)
        {
            return GetDevice().GetProvider(optimizationLevel);
        }


        /// <summary>
        /// Gets the CPU provider this DeviceType.
        /// </summary>
        /// <param name="deviceType">Type of the device.</param>
        /// <param name="optimizationLevel">The optimization level.</param>
        public static ExecutionProvider GetProvider(DeviceType deviceType, GraphOptimizationLevel optimizationLevel = GraphOptimizationLevel.ORT_DISABLE_ALL)
        {
            return GetDevice(deviceType).GetProvider(optimizationLevel);
        }


        /// <summary>
        /// Gets the CPU provider this DeviceType, DeviceId.
        /// </summary>
        /// <param name="deviceType">Type of the device.</param>
        /// <param name="deviceId">The device identifier.</param>
        /// <param name="optimizationLevel">The optimization level.</param>
        public static ExecutionProvider GetProvider(DeviceType deviceType, int deviceId, GraphOptimizationLevel optimizationLevel = GraphOptimizationLevel.ORT_DISABLE_ALL)
        {
            return GetDevice(deviceType, deviceId).GetProvider(optimizationLevel);
        }


        /// <summary>
        /// Gets the CPU provider for this Device.
        /// </summary>
        /// <param name="device">The device.</param>
        /// <param name="optimizationLevel">The optimization level.</param>
        public static ExecutionProvider GetProvider(this Device device, GraphOptimizationLevel optimizationLevel = GraphOptimizationLevel.ORT_DISABLE_ALL)
        {
            if (device == null)
                return default;
            else if (device.Type == DeviceType.NPU)
                return default;
            else if (device.Type == DeviceType.GPU)
                return default;

            return CreateProvider(optimizationLevel);
        }


        /// <summary>
        /// Gets the CPU provider.
        /// </summary>
        /// <param name="optimizationLevel">The optimization level.</param>
        /// <returns>ExecutionProvider.</returns>
        private static ExecutionProvider CreateProvider(GraphOptimizationLevel optimizationLevel = GraphOptimizationLevel.ORT_DISABLE_ALL)
        {
            return new ExecutionProvider(ProviderName, OrtMemoryInfo.DefaultInstance, configuration =>
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
