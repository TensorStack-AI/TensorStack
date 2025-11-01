namespace TensorStack.Common.Image
{
    public readonly struct ImagePixel
    {
        public readonly float R;
        public readonly float G;
        public readonly float B;
        public readonly float A;

        public ImagePixel(float r, float g, float b, float a)
        {
            R = r;
            G = g;
            B = b;
            A = a;
        }
    }
}
