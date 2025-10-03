// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System.Threading.Tasks;

namespace TensorStack.Audio
{
    public interface IAudioService
    {
        Task<AudioInfo> GetAudioInfoAsync(string filename);
    }
}
