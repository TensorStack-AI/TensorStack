using System;
using System.Windows.Media.Imaging;
using TensorStack.Audio;
using TensorStack.Common;
using TensorStack.Common.Tensor;
using TensorStack.Image;
using TensorStack.Video;

namespace TensorStack.WPF.Controls
{
    public record MediaImportEventArgs
    {
        public MediaImportEventArgs(MediaType mediaType, string mediaFile)
        {
            MediaType = mediaType;
            MediaFile = mediaFile;
        }

        public MediaImportEventArgs(string mediaFile, BitmapSource bitmapSource)
            : this(MediaType.Image, mediaFile)
        {
            Width = bitmapSource.PixelWidth;
            Height = bitmapSource.PixelHeight;
        }

        public MediaImportEventArgs(string mediaFile, ImageInput imageInput)
            : this(MediaType.Image, mediaFile)
        {
            Width = imageInput.Width;
            Height = imageInput.Height;
        }

        public MediaImportEventArgs(string mediaFile, AudioInputStream audioInput)
         : this(MediaType.Audio, mediaFile)
        {
            SampleRate = audioInput.SampleRate;
            Duration = audioInput.Duration;
        }

        public MediaImportEventArgs(string mediaFile, VideoInputStream videoStream)
            : this(MediaType.Video, mediaFile)
        {
            Width = videoStream.Width;
            Height = videoStream.Height;
            FrameRate = videoStream.FrameRate;
            FrameCount = videoStream.FrameCount;
            Duration = videoStream.Duration;
            Thumbnail = videoStream.Thumbnail;
        }


        public string MediaFile { get; init; }
        public MediaType MediaType { get; init; }
        public int Width { get; init; }
        public int Height { get; init; }
        public float FrameRate { get; init; }
        public int FrameCount { get; init; }
        public int SampleRate { get; init; }
        public TimeSpan Duration { get; init; }
        public ImageTensor Thumbnail { get; init; }
    }
}
