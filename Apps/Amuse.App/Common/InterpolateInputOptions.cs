using System.Runtime.CompilerServices;
using TensorStack.WPF;

namespace Amuse.App.Common
{
    public sealed record InterpolateInputOptions : BaseRecord
    {
        private int _multiplier;

        public int Multiplier
        {
            get { return _multiplier; }
            set { SetProperty(ref _multiplier, value); }
        }

        public bool Equals(InterpolateInputOptions other) => ReferenceEquals(this, other);
        public override int GetHashCode() => RuntimeHelpers.GetHashCode(this);
    }
}
