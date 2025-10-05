namespace TensorStack.Providers
{
    public record Device
    {
        public int DeviceId => PerformanceIndex; // TODO:
        public string Name { get; init; }
        public DeviceType Type { get; init; }
        public int Memory { get; init; }
        public int MemoryGB => Memory / 1000;
        public int AdapterIndex { get; init; }
        public int PerformanceIndex { get; init; }

        public int HardwareID { get; init; }
        public int HardwareLUID { get; init; }
        public int HardwareVendorId { get; init; }
        public string HardwareVendor { get; init; }
    }
}
