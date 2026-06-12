using TensorStack.WPF;

namespace Amuse.App.Common
{
    public sealed class AccessTokenModel : BaseModel
    {
        public string Name { get; set; }
        public string Token { get; set; }
        public string Domain { get; set; }
    }
}
