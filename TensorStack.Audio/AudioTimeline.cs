// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System;
using System.Collections.Generic;

namespace TensorStack.Audio
{
    public class AudioTimeline
    {
        public AudioTimeline(TimeSpan duration)
        {
            Duration = duration;
            Overlap = TimeSpan.FromMilliseconds(10);
        }

        public int SampleRate { get; init; } = 48000;
        public string Codec { get; init; } = "aac";
        public string Bitrate { get; init; } = "192k";
        public int Channels { get; init; } = 2;
        public string ChannelStr => Channels > 1 ? "stereo" : "mono";
        public TimeSpan Overlap { get; init; }
        public TimeSpan Duration { get; init; }
        public List<AudioSegment> Segments { get; init; } = [];
    }


    public record AudioSegment(string Source, TimeSpan Start, TimeSpan Duration, TimeSpan Position)
    {
        public bool IsFirst { get; set; }
        public bool IsLast { get; set; }
        public string FileName { get; set; }
    }
}
