# TensorStack.Providers.DML

The `Provider` class in the `TensorStack.Providers.DML` namespace gives you a simple, unified way to access available compute devices and create ONNX Runtime execution providers for them.

---

### Basic Initialization

```csharp
using TensorStack.Providers;

// Initialize the provider once (recommended at startup)
Provider.Initialize();

// Or initialize with custom environment options
var options = new EnvironmentCreationOptions
{
    // Example: configure threading, logging, or memory behavior
};
Provider.Initialize(options);
```

---

### Listing Available Devices

```csharp
// Get all available devices (CPU, GPU, NPU, etc.)
var devices = Provider.GetDevices();

foreach (var device in devices)
{
    Console.WriteLine($"{device.Type} - ID: {device.DeviceId}");
}
```

---

### Getting a Specific Device

```csharp
// Get the default CPU device
var cpu = Provider.GetDevice(DeviceType.CPU);

// Get a specific GPU device (e.g., device ID 0)
var gpu = Provider.GetDevice(DeviceType.GPU, 0);
```

---

### Creating an Execution Provider

```csharp
using Microsoft.ML.OnnxRuntime;

// Create an ExecutionProvider for the CPU
var cpuProvider = Provider.GetProvider(GraphOptimizationLevel.ORT_ENABLE_ALL);

// Create an ExecutionProvider for a specific device type
var gpuProvider = Provider.GetProvider(DeviceType.GPU, GraphOptimizationLevel.ORT_ENABLE_ALL);

// Create an ExecutionProvider for a specific device and ID
var customProvider = Provider.GetProvider(DeviceType.CPU, 0, GraphOptimizationLevel.ORT_ENABLE_EXTENDED);
```

---

### Full Example

```csharp
using System;
using TensorStack.Providers;
using Microsoft.ML.OnnxRuntime;

class Program
{
    static void Main()
    {
        // Initialize TensorStack provider system
        Provider.Initialize();

        // Choose the best CPU device
        var device = Provider.GetDevice();

        // Create an execution provider optimized for inference
        var provider = Provider.GetProvider(device, GraphOptimizationLevel.ORT_ENABLE_ALL);

        Console.WriteLine($"Using device: {device.Type} (ID: {device.DeviceId})");
    }
}
```

---

### Notes
- `Provider.Initialize()` only runs once per process; additional calls are ignored.
- `GetDevices()` automatically initializes the provider if it hasn’t been called yet.
- By default, this provider wraps the **CPU execution provider**.
- GPU and NPU logic are placeholders and can be extended in your own implementations.

--- 
