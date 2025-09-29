// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System.ComponentModel.DataAnnotations;
using TensorStack.Common.Tensor;
using TensorStack.TextGeneration.Common;

namespace TensorStack.TextGeneration.Pipelines.Whisper
{
    public record WhisperOptions : GenerateOptions
    {
        public TaskType Task { get; set; }
        public LanguageType Language { get; set; }
        public Tensor<float> AudioData { get; set; }
    }


    public record WhisperSearchOptions : WhisperOptions
    {
        public WhisperSearchOptions() { }
        public WhisperSearchOptions(WhisperOptions options) : base(options) { }
    }


    public enum TaskType
    {
        None = 0,
        Translate = 50358,
        Transcribe = 50359
    }

    public enum LanguageType
    {
        [Display(Name = "English")]
        EN = 50259,

        [Display(Name = "Chinese")]
        ZH = 50260,

        [Display(Name = "German")]
        DE = 50261,

        [Display(Name = "Spanish")]
        ES = 50262,

        [Display(Name = "Russian")]
        RU = 50263,

        [Display(Name = "Korean")]
        KO = 50264,

        [Display(Name = "French")]
        FR = 50265,

        [Display(Name = "Japanese")]
        JA = 50266,

        [Display(Name = "Portuguese")]
        PT = 50267,

        [Display(Name = "Turkish")]
        TR = 50268,

        [Display(Name = "Polish")]
        PL = 50269,

        [Display(Name = "Catalan")]
        CA = 50270,

        [Display(Name = "Dutch")]
        NL = 50271,

        [Display(Name = "Arabic")]
        AR = 50272,

        [Display(Name = "Swedish")]
        SV = 50273,

        [Display(Name = "Italian")]
        IT = 50274,

        [Display(Name = "Indonesian")]
        ID = 50275,

        [Display(Name = "Hindi")]
        HI = 50276,

        [Display(Name = "Finnish")]
        FI = 50277,

        [Display(Name = "Vietnamese")]
        VI = 50278,

        [Display(Name = "Hebrew")]
        HE = 50279,

        [Display(Name = "Ukrainian")]
        UK = 50280,

        [Display(Name = "Greek")]
        EL = 50281,

        [Display(Name = "Malay")]
        MS = 50282,

        [Display(Name = "Czech")]
        CS = 50283,

        [Display(Name = "Romanian")]
        RO = 50284,

        [Display(Name = "Danish")]
        DA = 50285,

        [Display(Name = "Hungarian")]
        HU = 50286,

        [Display(Name = "Tamil")]
        TA = 50287,

        [Display(Name = "Norwegian")]
        NO = 50288,

        [Display(Name = "Thai")]
        TH = 50289,

        [Display(Name = "Urdu")]
        UR = 50290,

        [Display(Name = "Croatian")]
        HR = 50291,

        [Display(Name = "Bulgarian")]
        BG = 50292,

        [Display(Name = "Lithuanian")]
        LT = 50293,

        [Display(Name = "Latin")]
        LA = 50294,

        [Display(Name = "Maori")]
        MI = 50295,

        [Display(Name = "Malayalam")]
        ML = 50296,

        [Display(Name = "Welsh")]
        CY = 50297,

        [Display(Name = "Slovak")]
        SK = 50298,

        [Display(Name = "Telugu")]
        TE = 50299,

        [Display(Name = "Persian")]
        FA = 50300,

        [Display(Name = "Latvian")]
        LV = 50301,

        [Display(Name = "Bengali")]
        BN = 50302,

        [Display(Name = "Serbian")]
        SR = 50303,

        [Display(Name = "Azerbaijani")]
        AZ = 50304,

        [Display(Name = "Slovenian")]
        SL = 50305,

        [Display(Name = "Kannada")]
        KN = 50306,

        [Display(Name = "Estonian")]
        ET = 50307,

        [Display(Name = "Macedonian")]
        MK = 50308,

        [Display(Name = "Breton")]
        BR = 50309,

        [Display(Name = "Basque")]
        EU = 50310,

        [Display(Name = "Icelandic")]
        IS = 50311,

        [Display(Name = "Armenian")]
        HY = 50312,

        [Display(Name = "Nepali")]
        NE = 50313,

        [Display(Name = "Mongolian")]
        MN = 50314,

        [Display(Name = "Bosnian")]
        BS = 50315,

        [Display(Name = "Kazakh")]
        KK = 50316,

        [Display(Name = "Albanian")]
        SQ = 50317,

        [Display(Name = "Swahili")]
        SW = 50318,

        [Display(Name = "Galician")]
        GL = 50319,

        [Display(Name = "Marathi")]
        MR = 50320,

        [Display(Name = "Punjabi")]
        PA = 50321,

        [Display(Name = "Sinhala")]
        SI = 50322,

        [Display(Name = "Khmer")]
        KM = 50323,

        [Display(Name = "Shona")]
        SN = 50324,

        [Display(Name = "Yoruba")]
        YO = 50325,

        [Display(Name = "Somali")]
        SO = 50326,

        [Display(Name = "Afrikaans")]
        AF = 50327,

        [Display(Name = "Occitan")]
        OC = 50328,

        [Display(Name = "Georgian")]
        KA = 50329,

        [Display(Name = "Belarusian")]
        BE = 50330,

        [Display(Name = "Tajik")]
        TG = 50331,

        [Display(Name = "Sindhi")]
        SD = 50332,

        [Display(Name = "Gujarati")]
        GU = 50333,

        [Display(Name = "Amharic")]
        AM = 50334,

        [Display(Name = "Yiddish")]
        YI = 50335,

        [Display(Name = "Lao")]
        LO = 50336,

        [Display(Name = "Uzbek")]
        UZ = 50337,

        [Display(Name = "Faroese")]
        FO = 50338,

        [Display(Name = "Haitian")]
        HT = 50339,

        [Display(Name = "Pashto")]
        PS = 50340,

        [Display(Name = "Turkmen")]
        TK = 50341,

        [Display(Name = "Norwegian Nynorsk")]
        NN = 50342,

        [Display(Name = "Maltese")]
        MT = 50343,

        [Display(Name = "Sanskrit")]
        SA = 50344,

        [Display(Name = "Luxembourgish")]
        LB = 50345,

        [Display(Name = "Burmese")]
        MY = 50346,

        [Display(Name = "Tibetan")]
        BO = 50347,

        [Display(Name = "Tagalog")]
        TL = 50348,

        [Display(Name = "Malagasy")]
        MG = 50349,

        [Display(Name = "Assamese")]
        AS = 50350,

        [Display(Name = "Tatar")]
        TT = 50351,

        [Display(Name = "Hawaiian")]
        HAW = 50352,

        [Display(Name = "Lingala")]
        LN = 50353,

        [Display(Name = "Hausa")]
        HA = 50354,

        [Display(Name = "Bashkir")]
        BA = 50355,

        [Display(Name = "Javanese")]
        JW = 50356,

        [Display(Name = "Sundanese")]
        SU = 50357
    }

}
