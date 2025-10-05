using Microsoft.ML.OnnxRuntime;
using System.Collections.Generic;
using TensorStack.Common;

namespace TensorStack.Providers
{
    public static class Provider
    {
        private const string DMLProviderName = "DMLExecutionProvider";
        private static IReadOnlyList<Device> _devices;

        /// <summary>
        /// Initializes the Provider with the specified environment options.
        /// </summary>
        /// <param name="environmentOptions">The environment options.</param>
        public static void Initialize(EnvironmentCreationOptions environmentOptions)
        {
            Devices.Initialize(environmentOptions);
            GetDevices();
        }


        /// <summary>
        /// Gets the DirectML devices.
        /// </summary>
        public static IReadOnlyList<Device> GetDevices()
        {
            _devices ??= Devices.GetDevices(DMLProviderName);
            return _devices;
        }


        /// <summary>
        /// Gets the DirectML provider.
        /// </summary>
        /// <param name="device">The device.</param>
        /// <param name="optimizationLevel">The optimization level.</param>
        public static ExecutionProvider GetProvider(Device device, GraphOptimizationLevel optimizationLevel = GraphOptimizationLevel.ORT_DISABLE_ALL)
        {
            return GetProvider(device.DeviceId, optimizationLevel);
        }


        /// <summary>
        /// Gets the DirectML provider.
        /// </summary>
        /// <param name="deviceId">The device identifier.</param>
        /// <param name="optimizationLevel">The optimization level.</param>
        /// <returns>ExecutionProvider.</returns>
        public static ExecutionProvider GetProvider(int deviceId, GraphOptimizationLevel optimizationLevel = GraphOptimizationLevel.ORT_DISABLE_ALL)
        {
            var memoryInfo = new OrtMemoryInfo(OrtMemoryInfo.allocatorCPU, OrtAllocatorType.DeviceAllocator, deviceId, OrtMemType.Default);
            return new ExecutionProvider(DMLProviderName, memoryInfo, configuration =>
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
    }
}
