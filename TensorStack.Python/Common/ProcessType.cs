namespace TensorStack.Python.Common
{
    public enum ProcessType
    {
        TextToImage = 100,
        ImageToImage = 101,
        ImageEdit = 102,
        ImageInpaint = 103,

        ControlNetImage = 200,
        ControlNetImageToImage = 201,

        TextToVideo = 300,
        ImageToVideo = 301,
        VideoToVideo = 302,
    }
}
