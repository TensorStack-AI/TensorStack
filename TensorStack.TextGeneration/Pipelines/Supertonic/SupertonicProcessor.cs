using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Text.RegularExpressions;
using TensorStack.Common.Tensor;

namespace TensorStack.TextGeneration.Pipelines.Supertonic
{
    /// <summary>
    /// Handle input text & voice styles for Supertonic.
    /// </summary>
    public class SupertonicProcessor
    {
        private readonly IReadOnlyDictionary<int, long> _indexer;
        private readonly IReadOnlyDictionary<string, VoiceStyle> _voiceStyles;

        /// <summary>
        /// Initializes a new instance of the <see cref="SupertonicProcessor"/> class.
        /// </summary>
        /// <param name="indexerPath">The indexer path.</param>
        /// <param name="voiceStylePath">The voice style path.</param>
        /// <exception cref="System.Exception">No valid indexer file found</exception>
        /// <exception cref="System.Exception">No valid voice files found</exception>
        public SupertonicProcessor(string indexerPath, string voiceStylePath)
        {
            _indexer = LoadIndexer(indexerPath);
            _voiceStyles = LoadVoiceStyles(voiceStylePath);
            if (_indexer.Count == 0)
                throw new Exception("No valid indexer file found");
            if (_voiceStyles.Count == 0)
                throw new Exception("No valid voice files found");
        }

        /// <summary>
        /// Gets the voice styles.
        /// </summary>
        public IEnumerable<string> VoiceStyles => _voiceStyles.Keys;


        /// <summary>
        /// Gets the TextIds in processable chunks.
        /// </summary>
        /// <param name="textInput">The text input.</param>
        /// <returns>List&lt;Tensor&lt;System.Int64&gt;&gt;.</returns>
        public List<Tensor<long>> GetTextIds(string textInput)
        {
            var textInputChunks = new List<Tensor<long>>();
            foreach (var textInputChunk in ChunkText(textInput))
            {
                textInputChunks.Add(GetTextIdsInternal(textInputChunk));
            }
            return textInputChunks;
        }


        /// <summary>
        /// Gets the voice style.
        /// </summary>
        /// <param name="styleName">Name of the style.</param>
        /// <returns>VoiceStyle.</returns>
        public VoiceStyle GetVoiceStyle(string styleName)
        {
            if (string.IsNullOrEmpty(styleName) || !_voiceStyles.ContainsKey(styleName))
                return _voiceStyles.Values.First();

            return _voiceStyles[styleName];
        }


        /// <summary>
        /// Gets the textds.
        /// </summary>
        /// <param name="textInput">The text input.</param>
        private Tensor<long> GetTextIdsInternal(string textInput)
        {
            var processedText = PreprocessText(textInput);
            var unicodeVals = TextToUnicodeValues(processedText);
            var textIds = new Tensor<long>([1, processedText.Length]);
            for (int j = 0; j < unicodeVals.Length; j++)
            {
                if (_indexer.TryGetValue(unicodeVals[j], out long val))
                {
                    textIds[0, j] = val;
                }
            }
            return textIds;
        }


        /// <summary>
        /// Split text input into processable chunks.
        /// </summary>
        /// <param name="text">The text.</param>
        /// <param name="maxLen">The maximum length.</param>
        private static List<string> ChunkText(string text, int maxLen = 300)
        {
            var chunks = new List<string>();

            // Split by paragraph (two or more newlines)
            var paragraphRegex = new Regex(@"\n\s*\n+");
            var paragraphs = paragraphRegex.Split(text.Trim())
                .Select(p => p.Trim())
                .Where(p => !string.IsNullOrEmpty(p))
                .ToList();

            // Split by sentence boundaries, excluding abbreviations
            var sentenceRegex = new Regex(@"(?<!Mr\.|Mrs\.|Ms\.|Dr\.|Prof\.|Sr\.|Jr\.|Ph\.D\.|etc\.|e\.g\.|i\.e\.|vs\.|Inc\.|Ltd\.|Co\.|Corp\.|St\.|Ave\.|Blvd\.)(?<!\b[A-Z]\.)(?<=[.!?])\s+");

            foreach (var paragraph in paragraphs)
            {
                var sentences = sentenceRegex.Split(paragraph);
                string currentChunk = "";

                foreach (var sentence in sentences)
                {
                    if (string.IsNullOrEmpty(sentence)) continue;

                    if (currentChunk.Length + sentence.Length + 1 <= maxLen)
                    {
                        if (!string.IsNullOrEmpty(currentChunk))
                        {
                            currentChunk += " ";
                        }
                        currentChunk += sentence;
                    }
                    else
                    {
                        if (!string.IsNullOrEmpty(currentChunk))
                        {
                            chunks.Add(currentChunk.Trim());
                        }
                        currentChunk = sentence;
                    }
                }

                if (!string.IsNullOrEmpty(currentChunk))
                {
                    chunks.Add(currentChunk.Trim());
                }
            }

            // If no chunks were created, return the original text
            if (chunks.Count == 0)
            {
                chunks.Add(text.Trim());
            }

            return chunks;
        }


        /// <summary>
        /// Convert Texts to unicode values.
        /// </summary>
        /// <param name="text">The text.</param>
        private static int[] TextToUnicodeValues(string text)
        {
            return [.. text.Select(c => (int)c)];
        }


        /// <summary>
        /// Removes the emojis.
        /// </summary>
        /// <param name="text">The text.</param>
        private static string RemoveEmojis(string text)
        {
            var sb = new StringBuilder(text.Length);

            for (int i = 0; i < text.Length; i++)
            {
                int codePoint;

                // Surrogate pair?
                if (char.IsHighSurrogate(text[i]) && i + 1 < text.Length && char.IsLowSurrogate(text[i + 1]))
                {
                    codePoint = char.ConvertToUtf32(text[i], text[i + 1]);
                    i++; // skip low surrogate
                }
                else
                {
                    codePoint = text[i];
                }

                if (IsEmoji(codePoint))
                    continue; // skip

                // re-append character
                if (codePoint > 0xFFFF)
                    sb.Append(char.ConvertFromUtf32(codePoint));
                else
                    sb.Append((char)codePoint);
            }

            return sb.ToString();
        }


        /// <summary>
        /// Determines whether the specified code point is emoji.
        /// </summary>
        /// <param name="codePoint">The code point.</param>
        private static bool IsEmoji(int codePoint)
        {
            // Covers all major emoji blocks
            return
                (codePoint >= 0x1F000 && codePoint <= 0x1FAFF) || // primary emoji planes
                (codePoint >= 0x2600 && codePoint <= 0x27BF) || // dingbats & misc symbols
                (codePoint >= 0xFE00 && codePoint <= 0xFE0F) || // variation selectors
                (codePoint >= 0x1F1E6 && codePoint <= 0x1F1FF);   // flags (regional indicators)
        }


        /// <summary>
        /// Preprocesses the text.
        /// </summary>
        /// <param name="text">The text.</param>
        /// <returns>System.String.</returns>
        private static string PreprocessText(string text)
        {
            // TODO: Need advanced normalizer for better performance
            text = text.Normalize(NormalizationForm.FormKD);

            // FIXME: this should be fixed for non-English languages

            // Remove emojis (wide Unicode range)
            // C# doesn't support \u{...} syntax in regex, so we use character filtering instead
            text = RemoveEmojis(text);

            // Replace various dashes and symbols
            var replacements = new Dictionary<string, string>
            {
                {"–", "-"},      // en dash
                {"‑", "-"},      // non-breaking hyphen
                {"—", "-"},      // em dash
                {"¯", " "},      // macron
                {"_", " "},      // underscore
                {"\u201C", "\""},     // left double quote
                {"\u201D", "\""},     // right double quote
                {"\u2018", "'"},      // left single quote
                {"\u2019", "'"},      // right single quote
                {"´", "'"},      // acute accent
                {"`", "'"},      // grave accent
                {"[", " "},      // left bracket
                {"]", " "},      // right bracket
                {"|", " "},      // vertical bar
                {"/", " "},      // slash
                {"#", " "},      // hash
                {"→", " "},      // right arrow
                {"←", " "},      // left arrow
            };

            foreach (var kvp in replacements)
            {
                text = text.Replace(kvp.Key, kvp.Value);
            }

            // Remove combining diacritics // FIXME: this should be fixed for non-English languages
            text = Regex.Replace(text, @"[\u0302\u0303\u0304\u0305\u0306\u0307\u0308\u030A\u030B\u030C\u0327\u0328\u0329\u032A\u032B\u032C\u032D\u032E\u032F]", "");

            // Remove special symbols
            text = Regex.Replace(text, @"[♥☆♡©\\]", "");

            // Replace known expressions
            var exprReplacements = new Dictionary<string, string>
            {
                {"@", " at "},
                {"e.g.,", "for example, "},
                {"i.e.,", "that is, "},
            };

            foreach (var kvp in exprReplacements)
            {
                text = text.Replace(kvp.Key, kvp.Value);
            }

            // Fix spacing around punctuation
            text = Regex.Replace(text, @" ,", ",");
            text = Regex.Replace(text, @" \.", ".");
            text = Regex.Replace(text, @" !", "!");
            text = Regex.Replace(text, @" \?", "?");
            text = Regex.Replace(text, @" ;", ";");
            text = Regex.Replace(text, @" :", ":");
            text = Regex.Replace(text, @" '", "'");

            // Remove duplicate quotes
            while (text.Contains("\"\""))
            {
                text = text.Replace("\"\"", "\"");
            }
            while (text.Contains("''"))
            {
                text = text.Replace("''", "'");
            }
            while (text.Contains("``"))
            {
                text = text.Replace("``", "`");
            }

            // Remove extra spaces
            text = Regex.Replace(text, @"\s+", " ").Trim();

            // If text doesn't end with punctuation, quotes, or closing brackets, add a period
            if (!Regex.IsMatch(text, @"[.!?;:,'\u0022\u201C\u201D\u2018\u2019)\]}…。」』】〉》›»]$"))
            {
                text += ".";
            }

            return text;
        }


        /// <summary>
        /// Creates a tensor from VoiceData.
        /// </summary>
        /// <param name="styleData">The style data.</param>
        /// <returns>Tensor&lt;System.Single&gt;.</returns>
        private static Tensor<float> CreateTensor(VoiceDataJson styleData)
        {
            var idx = 0;
            var dims = styleData.Dimensions;
            var tensor = new Tensor<float>(dims);
            for (int b = 0; b < dims[0]; b++)
                for (int d = 0; d < dims[1]; d++)
                    for (int t = 0; t < dims[2]; t++)
                        tensor.Memory.Span[idx++] = styleData.Data[b][d][t];
            return tensor;
        }


        /// <summary>
        /// Loads the voice styles.
        /// </summary>
        /// <param name="stylePath">The style path.</param>
        private static Dictionary<string, VoiceStyle> LoadVoiceStyles(string stylePath)
        {
            var voiceStyles = new Dictionary<string, VoiceStyle>();
            foreach (var styleFile in Directory.EnumerateFiles(stylePath, "*.json", SearchOption.TopDirectoryOnly))
            {
                try
                {
                    using (var jsonReader = File.OpenRead(styleFile))
                    {
                        var voiceStyle = JsonSerializer.Deserialize<VoiceStyleJson>(jsonReader);
                        var globalTensor = CreateTensor(voiceStyle.Global);
                        var dropoutTensor = CreateTensor(voiceStyle.Dropout);
                        var name = Path.GetFileNameWithoutExtension(styleFile);
                        voiceStyles.Add(name, new VoiceStyle(name, globalTensor, dropoutTensor));
                    }

                }
                catch (Exception)
                {
                    // TODO:
                }
            }
            return voiceStyles;
        }


        /// <summary>
        /// Loads the indexer.
        /// </summary>
        /// <param name="indexerPath">The indexer path.</param>
        private static Dictionary<int, long> LoadIndexer(string indexerPath)
        {
            var indexer = new Dictionary<int, long>();
            using (var jsonReader = File.OpenRead(indexerPath))
            {
                var indexerArray = JsonSerializer.Deserialize<long[]>(jsonReader);
                for (int i = 0; i < indexerArray.Length; i++)
                {
                    indexer.Add(i, indexerArray[i]);
                }
                return indexer;
            }
        }


        private record VoiceStyleJson
        {
            [JsonPropertyName("style_ttl")]
            public VoiceDataJson Global { get; set; }

            [JsonPropertyName("style_dp")]
            public VoiceDataJson Dropout { get; set; }
        }


        private record VoiceDataJson
        {
            [JsonPropertyName("data")]
            public float[][][] Data { get; set; }

            [JsonPropertyName("dims")]
            public int[] Dimensions { get; set; }
        }
    }
}
