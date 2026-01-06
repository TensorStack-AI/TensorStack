// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
namespace TensorStack.Common
{
    public record Device
    {
        public int Id { get; init; }
        public int DeviceId { get; init; }
        public string Name { get; init; }
        public DeviceType Type { get; init; }
        public int Memory { get; init; }
        public int MemoryGB => Memory / 1024;

        public VendorType Vendor { get; init; }
        public int HardwareID { get; init; }
        public int HardwareLUID { get; init; }
        public int HardwareVendorId { get; init; }
        public string HardwareVendor { get; init; }
    }
}
