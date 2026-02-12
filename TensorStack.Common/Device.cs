// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System;

namespace TensorStack.Common
{
    public record Device
    {
        private int _memory;

        public int Id { get; init; }
        public int DeviceId { get; init; }
        public string Name { get; init; }
        public DeviceType Type { get; init; }
        public VendorType Vendor { get; init; }
        public int HardwareID { get; init; }
        public int HardwareLUID { get; init; }
        public int HardwareVendorId { get; init; }
        public string HardwareVendor { get; init; }
        public int Memory
        {
            get { return _memory; }
            init
            {
                _memory = value;
                MemoryGB = (int)Math.Round(_memory / 1024.0, 0, MidpointRounding.ToEven);
            }
        }
        public int MemoryGB { get; init; }
    }
}
