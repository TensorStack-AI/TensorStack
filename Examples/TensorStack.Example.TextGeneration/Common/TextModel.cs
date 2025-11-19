using TensorStack.WPF;

namespace TensorStack.Example.Common
{
    public class TextModel : BaseModel
    {
        public int Id { get; init; }
        public string Name { get; init; }
        public bool IsDefault { get; set; }
        public TextModelType Type { get; init; }
        public string Version { get; set; }
        public int MinLength { get; init; }
        public int MaxLength { get; init; }
        public string[] Prefixes { get; init; }
        public string Path { get; set; }
        public string[] UrlPaths { get; set; }
    }


    public enum TextModelType
    {
        Summary = 0,
        Phi3 = 1,
        Whisper = 2
    }
}
