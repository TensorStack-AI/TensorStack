// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
namespace TensorStack.Video
{
    public interface IVideoConfiguration
    {
        string FFmpegPath { get; set; }
        string FFprobePath { get; set; }
        string DirectoryTemp { get; set; }

        int ReadBuffer { get; set; }
        int WriteBuffer { get; set; }
        string VideoCodec { get; set; }
    }
}
