using System.Collections.Generic;

namespace Amuse.Common.Config
{
    public sealed record ClientConfig
    {
        public bool IsDebugMode { get; set; }
        public string ServerPath { get; set; }
        public ServerType ServerType { get; set; }
        public Dictionary<string, string> ServerVariables { get; set; }
    }
}
