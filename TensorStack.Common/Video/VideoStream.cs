using System.Collections.Generic;
using System.Threading;

namespace TensorStack.Common.Video
{
    public class VideoStream : IAsyncEnumerable<VideoFrame>
    {
        private readonly IAsyncEnumerable<VideoFrame> _stream;
        private readonly int _width;
        private readonly int _height;
        private readonly float _frameRate;
        private readonly int _frameCount;

        public VideoStream(IAsyncEnumerable<VideoFrame> stream, int frameCount, float frameRate, int width, int height)
        {
            _stream = stream;
            _frameCount = frameCount;
            _frameRate = frameRate;
            _width = width;
            _height = height;
        }

        public int Width => _width;
        public int Height => _height;
        public float FrameRate => _frameRate;
        public int FrameCount => _frameCount;

        public IAsyncEnumerator<VideoFrame> GetAsyncEnumerator(CancellationToken cancellationToken = default)
        {
            return _stream.GetAsyncEnumerator(cancellationToken);
        }
    }
}
