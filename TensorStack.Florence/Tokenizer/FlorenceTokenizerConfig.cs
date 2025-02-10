// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System.Collections.Generic;
using System.Text.Json.Serialization;

namespace TensorStack.Florence.Tokenizer
{
    public class FlorenceTokenizerConfig
    {
        public string Path { get; set; }

        [JsonPropertyName("add_prefix_space")] 
        public bool AddPrefixSpace { get; set; }


        [JsonPropertyName("added_tokens_decoder")] 
        public Dictionary<string, AddedToken> AddedTokensDecoder { get; set; }


        [JsonPropertyName("additional_special_tokens")] 
        public string[] AdditionalSpecialTokens { get; set; }


        [JsonPropertyName("bos_token")]
        public string BosToken { get; set; }


        [JsonPropertyName("clean_up_tokenization_spaces")] 
        public bool CleanUpTokenizationSpaces { get; set; }


        [JsonPropertyName("cls_token")] 
        public string ClsToken { get; set; }


        [JsonPropertyName("eos_token")]
        public string EosToken { get; set; }


        [JsonPropertyName("errors")] 
        public string Errors { get; set; }


        [JsonPropertyName("mask_token")] 
        public string MaskToken { get; set; }


        [JsonPropertyName("model_max_length")] 
        public int ModelMaxLength { get; set; }


        [JsonPropertyName("pad_token")]
        public string PadToken { get; set; }


        [JsonPropertyName("processor_class")] 
        public string ProcessorClass { get; set; }


        [JsonPropertyName("sep_token")] 
        public string SepToken { get; set; }


        [JsonPropertyName("tokenizer_class")]
        public string TokenizerClass { get; set; }


        [JsonPropertyName("trim_offsets")]
        public bool TrimOffsets { get; set; }


        [JsonPropertyName("unk_token")] 
        public string UnkToken { get; set; }
    }


    public class AddedToken
    {
        [JsonPropertyName("content")] 
        public string Content { get; set; }

        [JsonPropertyName("lstrip")] 
        public bool Lstrip { get; set; }

        [JsonPropertyName("normalized")] 
        public bool Normalized { get; set; }

        [JsonPropertyName("rstrip")] 
        public bool Rstrip { get; set; }

        [JsonPropertyName("single_word")] 
        public bool SingleWord { get; set; }

        [JsonPropertyName("special")]
        public bool Special { get; set; }
    }
}
