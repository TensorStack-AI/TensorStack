using Amuse.App.Views;
using System;
using TensorStack.Common;

namespace Amuse.App.Common
{
    public interface IHistoryItem
    {
        string Id { get; init; }
        int Version { get; init; }
        View Source { get; init; }
        MediaType MediaType { get; init; }
        DateTime Timestamp { get; init; }
        string Extension { get; init; }

        int Width { get; init; }
        int Height { get; init; }
        float FrameRate { get; init; }
        int FrameCount { get; init; }
        int SampleRate { get; init; }
        TimeSpan Duration { get; init; }

        string Model { get; init; }

        string FilePath { get; set; }
        string MediaPath { get; set; }
        string ThumbPath { get; set; }
        DateTime LastAccess { get; set; }
    }
}
