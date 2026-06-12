using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using TensorStack.Common;

namespace Amuse.App.Common
{
    public sealed record TextResult
    {
        public TextResult() { }
        public TextResult(params TextInput[] inputs)
        {
            Results.AddRange(inputs);
        }

        public List<TextInput> Results { get; set; } = [];
        public TextInput Result => Results.FirstOrDefault();


        public bool Equals(TextResult other) => ReferenceEquals(this, other);
        public override int GetHashCode() => RuntimeHelpers.GetHashCode(this);
    }
}
