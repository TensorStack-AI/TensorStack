using Amuse.App.Dialogs;
using Microsoft.Extensions.Logging;
using System;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using TensorStack.WPF.Services;

namespace Amuse.App.Services
{
    public sealed class MigrationService : IMigrationService
    {
        private readonly Settings _settings;
        private readonly ILogger<MigrationService> _logger;

        public MigrationService(Settings settings, ILogger<MigrationService> logger)
        {
            _logger = logger;
            _settings = settings;
        }


        public Task RunMigrationsAsync()
        {
            _settings.RunMigrations = true;
            return RunAutoMigrationsAsync();
        }


        public async Task RunAutoMigrationsAsync()
        {
            _logger.LogInformation("[RunMigrations] Checking migrations...");
            if (!_settings.RunMigrations)
            {
                _logger.LogInformation("[RunMigrations] Migrations not required.");
                return;
            }

            // Run required migrations
            if (IsMigrationRequired(_settings.DirectoryModel))
            {
                _logger.LogInformation("[RunMigrations] Application migrations found, Migrating...");
                RunMigrations(_settings.DirectoryModel, false);
                _logger.LogInformation("[RunMigrations] Application migrations complete.");
            }

            // Run HuggingFace required migrations
            if (IsHuggingFaceMigrationRequired(_settings.DirectoryModel))
            {
                _logger.LogInformation("[RunMigrations] HuggingFace migrations found, Migrating...");
                var isDefaultModelLocation = IsDefaultModelLocation();
                if (isDefaultModelLocation)
                {
                    // Automatic: HuggingFace migration if model directory is owned by Amuse
                    RunHuggingFaceMigrations(_settings.DirectoryModel, false);
                    _logger.LogInformation("[RunMigrations] HuggingFace migrations complete.");
                }
                else
                {
                    // Optional: HuggingFace migration if model directory is not owned by Amuse
                    if (await IsHuggingFaceMigrationsAllowed())
                    {
                        RunHuggingFaceMigrations(_settings.DirectoryModel, false);
                        _logger.LogInformation("[RunMigrations] HuggingFace migrations complete.");
                    }
                    else
                    {
                        _logger.LogInformation("[RunMigrations] HuggingFace migrations skipped by user request.");
                    }
                }
            }

            _settings.ScanModels();
            _settings.RunMigrations = false;
            await SettingsManager.SaveAsync(_settings);
            _logger.LogInformation("[RunMigrations] Migrations complete.");
        }


        private bool IsDefaultModelLocation()
        {
            return _settings.DirectoryModel.Equals(Path.Combine(App.DirectoryData, "Models"));
        }


        private bool IsMigrationRequired(string modelDirectory)
        {
            return RunMigrations(modelDirectory, true);
        }


        private bool IsHuggingFaceMigrationRequired(string modelDirectory)
        {
            return RunHuggingFaceMigrations(modelDirectory, true);
        }


        private async Task<bool> IsHuggingFaceMigrationsAllowed()
        {
            var dialog = DialogService.GetDialog<QuestionDialog>();
            return await dialog.ShowDialogAsync("Model Migrations", "Migrate Existing Models?", "Amuse needs to move & rename some models folders to operate correctly, if you use these models with other applications they may not function correctly after migration, if you are using other applications we suggest skipping migrations and adding models back manually", "Migrate", "Skip");
        }


        private bool RunMigrations(string modelDirectory, bool isReadOnly)
        {
            AmuseMigration[] amuseMigrations =
            [
                new AmuseMigration("Diffusion", "Audio\\Supertonic", "Supertonic-v2"),
                new AmuseMigration("Diffusion", "Audio\\Whisper-Tiny", "Whisper-Tiny"),
                new AmuseMigration("Diffusion", "Audio\\Whisper-Small", "Whisper-Small"),
                new AmuseMigration("Diffusion", "Audio\\Whisper-Medium", "Whisper-Medium"),
                new AmuseMigration("Diffusion", "Audio\\Whisper-Base", "Whisper-Base"),
                new AmuseMigration("Diffusion", "Audio\\Whisper-Large", "Whisper-Large"),

                new AmuseMigration("Diffusion", "StableDiffusion-amuse", "StableDiffusion-amuse"),
                new AmuseMigration("Diffusion", "StableDiffusion-XL-amuse", "StableDiffusion-XL-amuse"),
                new AmuseMigration("Diffusion", "StableCascade-amuse", "StableCascade-amuse"),
                new AmuseMigration("Diffusion", "StableDiffusion3-Medium-amuse", "StableDiffusion3-Medium-amuse"),
                new AmuseMigration("Diffusion", "StableDiffusion3.5-Medium-amuse", "StableDiffusion3.5-Medium-amuse"),
                new AmuseMigration("Diffusion", "StableDiffusion3.5-Large-amuse", "StableDiffusion3.5-Large-amuse"),
                new AmuseMigration("Diffusion", "StableDiffusion3.5-Large-Turbo-amuse", "StableDiffusion3.5-Large-Turbo-amuse"),
                new AmuseMigration("Diffusion", "FLUX.1-Schnell-amuse", "FLUX.1-Schnell-amuse"),
                new AmuseMigration("Diffusion", "FLUX.1-Dev-amuse", "FLUX.1-Dev-amuse"),
                new AmuseMigration("Diffusion", "FLUX.1-Kontext-amuse", "FLUX.1-Kontext-amuse"),
                new AmuseMigration("Diffusion", "Juggernaut-amuse", "Juggernaut-amuse"),
                new AmuseMigration("Diffusion", "Juggernaut-XL-amuse", "Juggernaut-XL-amuse"),
                new AmuseMigration("Diffusion", "Juggernaut-XL-Lightning-amuse", "Juggernaut-XL-Lightning-amuse"),
                new AmuseMigration("Diffusion", "Locomotion-amuse", "Locomotion-amuse"),
                new AmuseMigration("Diffusion", "Locomotion-Dreamshaper-amuse", "Locomotion-Dreamshaper-amuse"),
                new AmuseMigration("Diffusion", "Locomotion-epiCRealism-amuse", "Locomotion-epiCRealism-amuse"),
                new AmuseMigration("Diffusion", "Locomotion-Juggernaut-amuse", "Locomotion-Juggernaut-amuse"),
                new AmuseMigration("Diffusion", "Locomotion-CyberRealistic-amuse", "Locomotion-CyberRealistic-amuse"),
                new AmuseMigration("Diffusion", "Locomotion-RealisticVision-amuse", "Locomotion-RealisticVision-amuse"),
                new AmuseMigration("Diffusion", "Locomotion-Samaritan3D-amuse", "Locomotion-Samaritan3D-amuse"),
                new AmuseMigration("Diffusion", "Locomotion-ToonYou-amuse", "Locomotion-ToonYou-amuse"),
                new AmuseMigration("Diffusion", "Hyper-SD-amuse", "Hyper-SD-amuse"),
                new AmuseMigration("Diffusion", "SDXL-Lightning-amuse", "SDXL-Lightning-amuse"),
                new AmuseMigration("Diffusion", "Dreamshaper-amuse", "Dreamshaper-amuse"),
                new AmuseMigration("Diffusion", "Dreamshaper-LCM-amuse", "Dreamshaper-LCM-amuse"),
                new AmuseMigration("Diffusion", "Dreamshaper-XL-amuse", "Dreamshaper-XL-amuse"),
                new AmuseMigration("Diffusion", "Dreamshaper-XL-Lightning-amuse", "Dreamshaper-XL-Lightning-amuse"),
                new AmuseMigration("Diffusion", "AngraRealflex-LCM-amuse", "AngraRealflex-LCM-amuse"),
                new AmuseMigration("Diffusion", "Animatix-LCM-amuse", "Animatix-LCM-amuse"),
                new AmuseMigration("Diffusion", "Artifida-LCM-amuse", "Artifida-LCM-amuse"),
                new AmuseMigration("Diffusion", "ComicCraft-LCM-amuse", "ComicCraft-LCM-amuse"),
                new AmuseMigration("Diffusion", "CyberRealistic-LCM-amuse", "CyberRealistic-LCM-amuse"),
                new AmuseMigration("Diffusion", "DreamCrafter-LCM-amuse", "DreamCrafter-LCM-amuse"),
                new AmuseMigration("Diffusion", "Fluently-v4-LCM-amuse", "Fluently-v4-LCM-amuse"),
                new AmuseMigration("Diffusion", "Momentary-LCM-amuse", "Momentary-LCM-amuse"),
                new AmuseMigration("Diffusion", "PermissiveBeauty-LCM-amuse", "PermissiveBeauty-LCM-amuse"),
                new AmuseMigration("Diffusion", "PhotographerAlpha-LCM-amuse", "PhotographerAlpha-LCM-amuse"),
                new AmuseMigration("Diffusion", "Quick-LCM-amuse", "Quick-LCM-amuse"),
                new AmuseMigration("Diffusion", "Realistic-LCM-amuse", "Realistic-LCM-amuse"),
                new AmuseMigration("Diffusion", "RealModelBase-LCM-amuse", "RealModelBase-LCM-amuse"),
                new AmuseMigration("Diffusion", "SilversRealmix-LCM-amuse", "SilversRealmix-LCM-amuse"),
                new AmuseMigration("Diffusion", "CopaxTimeless-Lightning-amuse", "CopaxTimeless-Lightning-amuse"),
                new AmuseMigration("Diffusion", "CopaxTimeless-Turbo-amuse", "CopaxTimeless-Turbo-amuse"),
                new AmuseMigration("Diffusion", "DreamDiffusion-XL-amuse", "DreamDiffusion-XL-amuse"),
                new AmuseMigration("Diffusion", "Fluently-XL-Final-amuse", "Fluently-XL-Final-amuse"),
                new AmuseMigration("Diffusion", "HK-XL-amuse", "HK-XL-amuse"),
                new AmuseMigration("Diffusion", "iNiverseMix-XL-amuse", "iNiverseMix-XL-amuse"),
                new AmuseMigration("Diffusion", "JibMix-XL-Turbo-amuse", "JibMix-XL-Turbo-amuse"),
                new AmuseMigration("Diffusion", "MidgardPony-XL-amuse", "MidgardPony-XL-amuse"),
                new AmuseMigration("Diffusion", "PixelWave-XL-amuse", "PixelWave-XL-amuse"),
                new AmuseMigration("Diffusion", "RealisticVision-Lightning-amuse", "RealisticVision-Lightning-amuse"),
                new AmuseMigration("Diffusion", "RealisticVision-XL-amuse", "RealisticVision-XL-amuse"),
                new AmuseMigration("Diffusion", "Sleipnir-XL-amuse", "Sleipnir-XL-amuse"),
                new AmuseMigration("Diffusion", "TurboVision-XL-amuse", "TurboVision-XL-amuse"),
                new AmuseMigration("Diffusion", "ZavyChroma-XL-amuse", "ZavyChroma-XL-amuse"),
                new AmuseMigration("Diffusion", "AdamMix-XL-amuse", "AdamMix-XL-amuse"),
                new AmuseMigration("Diffusion", "AGXL-amuse", "AGXL-amuse"),
                new AmuseMigration("Diffusion", "Animagine-XL-amuse", "Animagine-XL-amuse"),
                new AmuseMigration("Diffusion", "AutismMix-XL-amuse", "AutismMix-XL-amuse"),
                new AmuseMigration("Diffusion", "BeyondImagination-XL-amuse", "BeyondImagination-XL-amuse"),
                new AmuseMigration("Diffusion", "Cinematix-XL-amuse", "Cinematix-XL-amuse"),
                new AmuseMigration("Diffusion", "CopaxTimeLess-XL-amuse", "CopaxTimeLess-XL-amuse"),
                new AmuseMigration("Diffusion", "CustomXL-Mirage-amuse", "CustomXL-Mirage-amuse"),
                new AmuseMigration("Diffusion", "CyberRealistic-XL-amuse", "CyberRealistic-XL-amuse"),
                new AmuseMigration("Diffusion", "DemonCore-XL-amuse", "DemonCore-XL-amuse"),
                new AmuseMigration("Diffusion", "FaserCore-XL-amuse", "FaserCore-XL-amuse"),
                new AmuseMigration("Diffusion", "Fenris-XL-amuse", "Fenris-XL-amuse"),
                new AmuseMigration("Diffusion", "Fidelis-XL-amuse", "Fidelis-XL-amuse"),
                new AmuseMigration("Diffusion", "Hassaku-XL-amuse", "Hassaku-XL-amuse"),
                new AmuseMigration("Diffusion", "Lugansk-XL-amuse", "Lugansk-XL-amuse"),
                new AmuseMigration("Diffusion", "MegaChonk-XL-amuse", "MegaChonk-XL-amuse"),
                new AmuseMigration("Diffusion", "Moxie-Fusion-XL-amuse", "Moxie-Fusion-XL-amuse"),
                new AmuseMigration("Diffusion", "MS-Real-XL-amuse", "MS-Real-XL-amuse"),
                new AmuseMigration("Diffusion", "PhotoArt-XL-amuse", "PhotoArt-XL-amuse"),
                new AmuseMigration("Diffusion", "Photonic-Fusion-XL-amuse", "Photonic-Fusion-XL-amuse"),
                new AmuseMigration("Diffusion", "QuadPipe-XL-amuse", "QuadPipe-XL-amuse"),
                new AmuseMigration("Diffusion", "Boltning-Lightning-amuse", "Boltning-Lightning-amuse"),
                new AmuseMigration("Diffusion", "Fenris-XL-Lightning-amuse", "Fenris-XL-Lightning-amuse"),
                new AmuseMigration("Diffusion", "Lightning-Fusion-XL-amuse", "Lightning-Fusion-XL-amuse"),
                new AmuseMigration("Diffusion", "PrefectPony-Lightning-amuse", "PrefectPony-Lightning-amuse"),
                new AmuseMigration("Diffusion", "Ragnarok-XL-amuse", "Ragnarok-XL-amuse"),
                new AmuseMigration("Diffusion", "Ratatoskr-XL-amuse", "Ratatoskr-XL-amuse"),
                new AmuseMigration("Diffusion", "RealArchvis-XL-amuse", "RealArchvis-XL-amuse"),
                new AmuseMigration("Diffusion", "RealDream-XL-amuse", "RealDream-XL-amuse"),
                new AmuseMigration("Diffusion", "RealitiesEdge-Lightning-amuse", "RealitiesEdge-Lightning-amuse"),
                new AmuseMigration("Diffusion", "RealitiesEdge-XL-amuse", "RealitiesEdge-XL-amuse"),
                new AmuseMigration("Diffusion", "RealSpice-XL-amuse", "RealSpice-XL-amuse"),
                new AmuseMigration("Diffusion", "Tempest-XL-amuse", "Tempest-XL-amuse"),
                new AmuseMigration("Diffusion", "Ultrium-XL-amuse", "Ultrium-XL-amuse"),
                new AmuseMigration("Diffusion", "WildCardX-XL-amuse", "WildCardX-XL-amuse"),
                new AmuseMigration("Diffusion", "Xi-XL-amuse", "Xi-XL-amuse"),
                new AmuseMigration("Diffusion", "Zaxious-XL-aumse", "Zaxious-XL-aumse"),
                new AmuseMigration("Diffusion", "AbsoluteReality_v181-amuse", "AbsoluteReality_v181-amuse"),
                new AmuseMigration("Diffusion", "AirtistPhoto-amuse", "AirtistPhoto-amuse"),
                new AmuseMigration("Diffusion", "AnalogMadness_v70-amuse", "AnalogMadness_v70-amuse"),
                new AmuseMigration("Diffusion", "AziibPixelMix-amuse", "AziibPixelMix-amuse"),
                new AmuseMigration("Diffusion", "aZovyaPhotoreal_v3-amuse", "aZovyaPhotoreal_v3-amuse"),
                new AmuseMigration("Diffusion", "Colorful_v80-amuse", "Colorful_v80-amuse"),
                new AmuseMigration("Diffusion", "CopaxTimeless-amuse", "CopaxTimeless-amuse"),
                new AmuseMigration("Diffusion", "DarkSushiMix-amuse", "DarkSushiMix-amuse"),
                new AmuseMigration("Diffusion", "EpicRealism_v5-amuse", "EpicRealism_v5-amuse"),
                new AmuseMigration("Diffusion", "Experience_V10-amuse", "Experience_V10-amuse"),
                new AmuseMigration("Diffusion", "Fluently-v4-amuse", "Fluently-v4-amuse"),
                new AmuseMigration("Diffusion", "Lyriel_v16-amuse", "Lyriel_v16-amuse"),
                new AmuseMigration("Diffusion", "MajicmixRealistic_v7-amuse", "MajicmixRealistic_v7-amuse"),
                new AmuseMigration("Diffusion", "Meinamix_V11-amuse", "Meinamix_V11-amuse"),
                new AmuseMigration("Diffusion", "NeverendingDream_v122-amuse", "NeverendingDream_v122-amuse"),
                new AmuseMigration("Diffusion", "Photon-amuse", "Photon-amuse"),
                new AmuseMigration("Diffusion", "RealCartoon3D-amuse", "RealCartoon3D-amuse"),
                new AmuseMigration("Diffusion", "Realdosmix-amuse", "Realdosmix-amuse"),
                new AmuseMigration("Diffusion", "RealisticVision_v6-amuse", "RealisticVision_v6-amuse"),
                new AmuseMigration("Diffusion", "RevAnimated_v2-amuse", "RevAnimated_v2-amuse"),
                new AmuseMigration("Diffusion", "RPG_v5-amuse", "RPG_v5-amuse"),
                new AmuseMigration("Diffusion", "sxzLuma-amuse", "sxzLuma-amuse"),
                new AmuseMigration("Diffusion", "ToonYou_v6-amuse", "ToonYou_v6-amuse"),
                new AmuseMigration("Diffusion", "UnstableIllusion-amuse", "UnstableIllusion-amuse"),
                new AmuseMigration("Diffusion", "Yesmix_v50-amuse", "Yesmix_v50-amuse"),
                new AmuseMigration("Diffusion", "StableDiffusion-Instruct-amuse", "StableDiffusion-Instruct-amuse"),

                // Obsolete
                new AmuseMigration("Diffusion", "sdxl-turbo-ryzen-ai", "sdxl-turbo-ryzen-ai"),
                new AmuseMigration("Diffusion", "stable-diffusion-1.5_io32_amdgpu", "stable-diffusion-1.5_io32_amdgpu"),
                new AmuseMigration("Diffusion", "stable-diffusion-xl-1.0_io32_amdgpu", "stable-diffusion-xl-1.0_io32_amdgpu"),
                new AmuseMigration("Diffusion", "dreamshaper-xl-lightning_io32_amdgpu", "dreamshaper-xl-lightning_io32_amdgpu"),
                new AmuseMigration("Diffusion", "sdxl-turbo_amdgpu", "sdxl-turbo_amdgpu"),
                new AmuseMigration("Diffusion", "stable-diffusion-3-medium_amdgpu", "stable-diffusion-3-medium_amdgpu"),
                new AmuseMigration("Diffusion", "stable-diffusion-3.5-medium_amdgpu", "stable-diffusion-3.5-medium_amdgpu"),
                new AmuseMigration("Diffusion", "stable-diffusion-3.5-large_amdgpu", "stable-diffusion-3.5-large_amdgpu"),
                new AmuseMigration("Diffusion", "stable-diffusion-3.5-large-turbo_amdgpu", "stable-diffusion-3.5-large-turbo_amdgpu"),
                new AmuseMigration("Diffusion", "FLUX.1-schnell_io32_amdgpu", "FLUX.1-schnell_io32_amdgpu"),
                new AmuseMigration("Diffusion", "FLUX.1-dev_io32_amdgpu", "FLUX.1-dev_io32_amdgpu")
            ];

            RenameMigration[] renameMigrations =
            [
                new RenameMigration("Upscale", "AnimeSharpV2 ESRGAN Soft 2x", "AnimeSharpV2-ESRGAN-Soft-2x"),
                new RenameMigration("Upscale", "AnimeSharpV2 MoSR Sharp 2x", "AnimeSharpV2-MoSR-Sharp-2x"),
                new RenameMigration("Upscale", "AnimeSharpV2 MoSR Soft 2x", "AnimeSharpV2-MoSR-Soft-2x"),
                new RenameMigration("Upscale", "AnimeSharpV2 RPLKSR Sharp 2x", "AnimeSharpV2-RPLKSR-Sharp-2x"),
                new RenameMigration("Upscale", "AnimeSharpV2 RPLKSR Soft 2x", "AnimeSharpV2-RPLKSR-Soft-2x"),
                new RenameMigration("Upscale", "AnimeSharpV3 2x", "AnimeSharpV3-2x"),
                new RenameMigration("Upscale", "APISR GRL GAN 4x", "APISR-GRL-GAN-4x"),
                new RenameMigration("Upscale", "APISR RRDB GAN 2x", "APISR-RRDB-GAN-2x"),
                new RenameMigration("Upscale", "BSRGAN 2x", "BSRGAN-2x"),
                new RenameMigration("Upscale", "Real ESRGAN 2x", "RealESRGAN-2x"),
                new RenameMigration("Upscale", "Real ESRGAN 4x", "RealESRGAN-4x"),
                new RenameMigration("Upscale", "RealESR General 4x", "RealESR-General-4x"),
                new RenameMigration("Upscale", "RealWebPhoto RGT 4x", "RealWebPhoto-RGT-4x"),
                new RenameMigration("Upscale", "Swin2SR Classical 2x", "Swin2SR-Classical-2x"),
                new RenameMigration("Upscale", "Swin2SR Classical 4x", "Swin2SR-Classical-4x"),
                new RenameMigration("Upscale", "Swin2SR RealWorld BSRGAN PSN 4x", "Swin2SR-RealWorld-BSRGAN-PSN-4x"),
                new RenameMigration("Upscale", "SwinIR BSRGAN 4x", "SwinIR-BSRGAN-4x"),
                new RenameMigration("Upscale", "UltraMix Smooth 4x", "UltraMix-Smooth-4x"),
                new RenameMigration("Upscale", "UltraSharp 4x", "UltraSharp-4x")
            ];

            HuggingFaceMigration[] huggingFaceMigrations =
            [
                new HuggingFaceMigration("AutoEncoder", "models--TensorStack--AutoEncoder", null),
                new HuggingFaceMigration("TextEncoder", "models--TensorStack--TextEncoder", null),
            ];


            string[] deleteMigrations =
            [
                Path.Combine(modelDirectory, "Audio"),
                Path.Combine(modelDirectory, "Upscale-amuse"),
                Path.Combine(modelDirectory, "ControlNet-amuse"),
                Path.Combine(modelDirectory, "FeatureExtractor-amuse")
            ];

            return RunMigrations(modelDirectory, renameMigrations, isReadOnly)
                || RunMigrations(modelDirectory, amuseMigrations, isReadOnly)
                || RunMigrations(modelDirectory, huggingFaceMigrations, isReadOnly)
                || RunMigrations(modelDirectory, deleteMigrations, isReadOnly);
        }


        private bool RunHuggingFaceMigrations(string modelDirectory, bool isReadOnly)
        {
            HuggingFaceMigration[] huggingFaceMigrations =
            [
                new HuggingFaceMigration("Diffusion", "models--stabilityai--stable-diffusion-xl-base-1.0", "stable-diffusion-xl-base-1.0" ),
                new HuggingFaceMigration("Diffusion", "models--stabilityai--stable-diffusion-3-medium-diffusers", "stable-diffusion-3-medium-diffusers" ),
                new HuggingFaceMigration("Diffusion", "models--stabilityai--stable-diffusion-3.5-medium", "stable-diffusion-3.5-medium" ),
                new HuggingFaceMigration("Diffusion", "models--stabilityai--stable-diffusion-3.5-large", "stable-diffusion-3.5-large" ),
                new HuggingFaceMigration("Diffusion", "models--stabilityai--stable-diffusion-3.5-large-turbo", "stable-diffusion-3.5-large-turbo" ),
                new HuggingFaceMigration("Diffusion", "models--Tongyi-MAI--Z-Image", "Z-Image" ),
                new HuggingFaceMigration("Diffusion", "models--Tongyi-MAI--Z-Image-Turbo", "Z-Image-Turbo" ),
                new HuggingFaceMigration("Diffusion", "models--Qwen--Qwen-Image-2512", "Qwen-Image-2512" ),
                new HuggingFaceMigration("Diffusion", "models--Qwen--Qwen-Image-Edit-2511", "Qwen-Image-Edit-2511" ),
                new HuggingFaceMigration("Diffusion", "models--lodestones--Chroma1-HD", "Chroma1-HD" ),
                new HuggingFaceMigration("Diffusion", "models--TensorStack--FLUX.1-schnell-ts", "FLUX.1-schnell" ),
                new HuggingFaceMigration("Diffusion", "models--TensorStack--FLUX.1-dev-ts", "FLUX.1-dev" ),
                new HuggingFaceMigration("Diffusion", "models--TensorStack--FLUX.1-Kontext-dev-ts", "FLUX.1-Kontext-dev" ),
                new HuggingFaceMigration("Diffusion", "models--black-forest-labs--FLUX.2-klein-4B", "FLUX.2-klein-4B" ),
                new HuggingFaceMigration("Diffusion", "models--black-forest-labs--FLUX.2-klein-base-4B", "FLUX.2-klein-base-4B" ),
                new HuggingFaceMigration("Diffusion", "models--black-forest-labs--FLUX.2-klein-9B", "FLUX.2-klein-9B" ),
                new HuggingFaceMigration("Diffusion", "models--black-forest-labs--FLUX.2-klein-base-9B", "FLUX.2-klein-base-9B" ),
                new HuggingFaceMigration("Diffusion", "models--Wan-AI--Wan2.1-T2V-1.3B-Diffusers", "Wan2.1-T2V-1.3B-Diffusers" ),
                new HuggingFaceMigration("Diffusion", "models--Wan-AI--Wan2.1-T2V-14B-Diffusers", "Wan2.1-T2V-14B-Diffusers" ),
                new HuggingFaceMigration("Diffusion", "models--Wan-AI--Wan2.1-I2V-14B-480P-Diffusers", "Wan2.1-I2V-14B-480P-Diffusers" ),
                new HuggingFaceMigration("Diffusion", "models--Wan-AI--Wan2.2-TI2V-5B-Diffusers", "Wan2.2-TI2V-5B-Diffusers" ),
                new HuggingFaceMigration("Diffusion", "models--Wan-AI--Wan2.2-T2V-A14B-Diffusers", "Wan2.2-T2V-A14B-Diffusers" ),
                new HuggingFaceMigration("Diffusion", "models--kandinskylab--Kandinsky-5.0-T2I-Lite-sft-Diffusers", "Kandinsky-5.0-T2I-Lite-sft-Diffusers" ),
                new HuggingFaceMigration("Diffusion", "models--kandinskylab--Kandinsky-5.0-I2I-Lite-sft-Diffusers", "Kandinsky-5.0-I2I-Lite-sft-Diffusers" ),
                new HuggingFaceMigration("Diffusion", "models--kandinskylab--Kandinsky-5.0-T2V-Lite-sft-5s-Diffusers", "Kandinsky-5.0-T2V-Lite-sft-5s-Diffusers" ),
                new HuggingFaceMigration("Diffusion", "models--kandinskylab--Kandinsky-5.0-I2V-Pro-sft-5s-Diffusers", "Kandinsky-5.0-I2V-Pro-sft-5s-Diffusers" ),
                new HuggingFaceMigration("Diffusion", "models--zai-org--CogVideoX-2b", "CogVideoX-2b" ),
                new HuggingFaceMigration("Diffusion", "models--zai-org--CogVideoX-5b", "CogVideoX-5b" ),
                new HuggingFaceMigration("Diffusion", "models--zai-org--CogVideoX-5b-I2V", "CogVideoX-5b-I2V" ),
                new HuggingFaceMigration("Diffusion", "models--zai-org--CogVideoX1.5-5B", "CogVideoX1.5-5B" ),
                new HuggingFaceMigration("Diffusion", "models--zai-org--CogVideoX1.5-5B-I2V", "CogVideoX1.5-5B-I2V" ),
                new HuggingFaceMigration("Diffusion", "models--Lightricks--LTX-Video", "LTX-Video" ),
                new HuggingFaceMigration("Diffusion", "models--Lightricks--LTX-2", "LTX-2" ),
                new HuggingFaceMigration("Diffusion", "models--dg845--LTX-2.3-Diffusers", "LTX-2.3-Diffusers" ),
                new HuggingFaceMigration("Diffusion", "models--dg845--LTX-2.3-Distilled-Diffusers", "LTX-2.3-Distilled-Diffusers" ),
                new HuggingFaceMigration("Diffusion", "models--Skywork--SkyReels-V2-DF-1.3B-540P-Diffusers", "SkyReels-V2-DF-1.3B-540P-Diffusers" ),
                new HuggingFaceMigration("Diffusion", "models--Skywork--SkyReels-V2-DF-14B-540P-Diffusers", "SkyReels-V2-DF-14B-540P-Diffusers" ),
                new HuggingFaceMigration("Diffusion", "models--BestWishYsh--Helios-Base", "Helios-Base" ),
                new HuggingFaceMigration("Diffusion", "models--BestWishYsh--Helios-Mid", "Helios-Mid" ),
                new HuggingFaceMigration("Diffusion", "models--BestWishYsh--Helios-Distilled", "Helios-Distilled" ),
                new HuggingFaceMigration("Diffusion", "models--ACE-Step--acestep-v15-xl-turbo-diffusers", "acestep-v15-xl-turbo-diffusers" ),

                new HuggingFaceMigration("LoraAdapter", "models--godtoldmetodoit--zitloras", "ZImage" ),
                new HuggingFaceMigration("LoraAdapter", "models--renderartist--Coloring-Book-Z-Image-Turbo-LoRA", "LTX20" ),
                new HuggingFaceMigration("LoraAdapter", "models--Lightricks--LTX-2-19b-LoRA-Camera-Control-Dolly-In", "LTX20" ),
                new HuggingFaceMigration("LoraAdapter", "models--Lightricks--LTX-2-19b-LoRA-Camera-Control-Dolly-Out", "LTX20" ),
                new HuggingFaceMigration("LoraAdapter", "models--Lightricks--LTX-2-19b-LoRA-Camera-Control-Dolly-Left", "LTX20" ),
                new HuggingFaceMigration("LoraAdapter", "models--Lightricks--LTX-2-19b-LoRA-Camera-Control-Dolly-Right", "LTX20" ),
                new HuggingFaceMigration("LoraAdapter", "models--Lightricks--LTX-2-19b-LoRA-Camera-Control-Jib-Down", "LTX20" ),
                new HuggingFaceMigration("LoraAdapter", "models--Lightricks--LTX-2-19b-LoRA-Camera-Control-Jib-Up", "LTX20" ),
                new HuggingFaceMigration("LoraAdapter", "models--Lightricks--LTX-2-19b-LoRA-Camera-Control-Static", "LTX20" ),
                new HuggingFaceMigration("LoraAdapter", "models--Lightricks--LTX-2-19b-IC-LoRA-Detailer", "LTX20" ),
                new HuggingFaceMigration("LoraAdapter", "models--ByteDance--Hyper-SD", "StableDiffusionXL" ),
                new HuggingFaceMigration("LoraAdapter", "models--fal--flux-2-klein-4b-spritesheet-lora", "Flux2" ),
                new HuggingFaceMigration("LoraAdapter", "models--fal--flux-2-klein-4B-background-remove-lora", "Flux2" ),
                new HuggingFaceMigration("LoraAdapter", "models--fal--flux-2-klein-4B-zoom-lora", "Flux2" ),
                new HuggingFaceMigration("LoraAdapter", "models--fal--flux-2-klein-4B-outpaint-lora", "Flux2" ),
                new HuggingFaceMigration("LoraAdapter", "models--fal--flux-klein-9b-virtual-tryon-lora", "Flux2" ),
                new HuggingFaceMigration("LoraAdapter", "models--dx8152--Flux2-Klein-9B-Consistency", "Flux2" ),
                new HuggingFaceMigration("LoraAdapter", "models--dx8152--Flux2-Klein-9B-Enhanced-Details", "Flux2" ),
                new HuggingFaceMigration("LoraAdapter", "models--reverentelusarca--flux2-klein-9b-4b-scribbly-doodle-lora", "Flux2" ),

                new HuggingFaceMigration("ControlNet", "models--InstantX--Qwen-Image-ControlNet-Union", "QwenImage\\Qwen-Image-ControlNet-Union" ),
                new HuggingFaceMigration("ControlNet", "models--InstantX--FLUX.1-dev-controlnet-canny", "Flux1\\FLUX.1-dev-controlnet-canny" ),
                new HuggingFaceMigration("ControlNet", "models--xinsir--controlnet-union-sdxl-1.0", "StableDiffusionXL\\controlnet-union-sdxl-1.0" ),
                new HuggingFaceMigration("ControlNet", "models--xinsir--controlnet-canny-sdxl-1.0", "StableDiffusionXL\\controlnet-canny-sdxl-1.0" ),
                new HuggingFaceMigration("ControlNet", "models--xinsir--controlnet-depth-sdxl-1.0", "StableDiffusionXL\\controlnet-depth-sdxl-1.0" ),
                new HuggingFaceMigration("ControlNet", "models--xinsir--controlnet-openpose-sdxl-1.0", "StableDiffusionXL\\controlnet-openpose-sdxl-1.0" ),
                new HuggingFaceMigration("ControlNet", "models--xinsir--controlnet-scribble-sdxl-1.0", "StableDiffusionXL\\controlnet-scribble-sdxl-1.0" ),
                new HuggingFaceMigration("ControlNet", "models--xinsir--controlnet-tile-sdxl-1.0", "StableDiffusionXL\\controlnet-tile-sdxl-1.0" ),
                new HuggingFaceMigration("ControlNet", "models--hlky--Z-Image-Turbo-Fun-Controlnet-Union", "ZImage\\Z-Image-Turbo-Fun-Controlnet-Union" ),
            ];

            return RunMigrations(modelDirectory, huggingFaceMigrations, isReadOnly);
        }


        private bool RunMigrations(string modelDirectory, string[] migrations, bool isReadOnly)
        {
            foreach (var directory in migrations)
            {
                try
                {
                    if (Directory.Exists(directory))
                    {
                        if (isReadOnly)
                            return true;

                        Directory.Delete(directory, true);
                    }
                }
                catch (Exception ex)
                {
                    _logger.LogError("[DeleteMigration] Directory: {directory}, Error: {message}", directory, ex.Message);
                }
            }
            return false;
        }


        private bool RunMigrations(string modelDirectory, RenameMigration[] migrations, bool isReadOnly)
        {
            foreach (var migration in migrations)
            {
                try
                {
                    var migrationSource = Path.Combine(modelDirectory, migration.Type, migration.OldName);
                    if (!Directory.Exists(migrationSource))
                        continue;

                    var migrationDirectory = new DirectoryInfo(migrationSource);
                    if (migrationDirectory == null)
                        continue;

                    if (!migrationDirectory.Exists)
                        continue;

                    if (isReadOnly)
                        return true;

                    _logger.LogInformation("[RenameMigration] Type: {type}, Name: {oldName}, NewName: {newName}", migration.Type, migration.OldName, migration.NewName);
                    migrationDirectory.MoveTo(Path.Combine(modelDirectory, migration.Type, migration.NewName));
                }
                catch (Exception ex)
                {
                    _logger.LogError("[RenameMigration] Type: {type}, Name: {oldName}, Error: {message}", migration.Type, migration.OldName, ex.Message);
                }
            }
            return false;
        }


        private bool RunMigrations(string modelDirectory, AmuseMigration[] migrations, bool isReadOnly)
        {
            foreach (var migration in migrations)
            {
                try
                {
                    var migrationSource = Path.Combine(modelDirectory, migration.OldName);
                    if (!Directory.Exists(migrationSource))
                        continue;

                    var migrationDirectory = new DirectoryInfo(migrationSource);
                    if (migrationDirectory == null)
                        continue;

                    if (isReadOnly)
                        return true;

                    _logger.LogInformation("[AmuseMigration] Type: {type}, Name: {oldName}, NewName: {newName}", migration.Type, migration.OldName, migration.NewName);
                    ProcessDirectory(migrationSource, Path.Combine(modelDirectory, migration.Type, migration.NewName));
                    foreach (var subdirectory in migrationDirectory.GetDirectories())
                    {
                        ProcessDirectory(subdirectory.FullName, Path.Combine(modelDirectory, migration.Type, migration.NewName, subdirectory.Name));
                    }

                    if (Directory.Exists(migrationSource))
                        Directory.Delete(migrationSource, true);
                }
                catch (Exception ex)
                {
                    _logger.LogError("[AmuseMigration] Type: {type}, Name: {oldName}, Error: {message}", migration.Type, migration.OldName, ex.Message);
                }
            }
            return false;
        }


        private bool RunMigrations(string modelDirectory, HuggingFaceMigration[] migrations, bool isReadOnly)
        {
            foreach (var migration in migrations)
            {
                try
                {
                    var migrationSource = Path.Combine(modelDirectory, migration.OldName);
                    if (!Directory.Exists(migrationSource))
                        continue;

                    var migrationDirectory = GetSnapshot(migrationSource);
                    if (migrationDirectory == null)
                        continue;

                    if (isReadOnly)
                        return true;

                    _logger.LogInformation("[HuggingFaceMigration] Type: {type}, Name: {oldName}, NewName: {newName}", migration.Type, migration.OldName, migration.NewName);
                    if (string.IsNullOrEmpty(migration.NewName))
                    {
                        foreach (var subdirectory in migrationDirectory.GetDirectories())
                        {
                            ProcessDirectory(subdirectory.FullName, Path.Combine(modelDirectory, migration.Type, subdirectory.Name));
                        }
                    }
                    else
                    {
                        ProcessDirectory(migrationDirectory.FullName, Path.Combine(modelDirectory, migration.Type, migration.NewName));
                        foreach (var subdirectory in migrationDirectory.GetDirectories())
                        {
                            ProcessDirectory(subdirectory.FullName, Path.Combine(modelDirectory, migration.Type, migration.NewName, subdirectory.Name));
                        }
                    }

                    if (Directory.Exists(migrationSource))
                        Directory.Delete(migrationSource, true);
                }
                catch (Exception ex)
                {
                    _logger.LogError("[HuggingFaceMigration] Type: {type}, Name: {oldName}, Error: {message}", migration.Type, migration.OldName, ex.Message);
                }
            }
            return false;
        }


        private static void ProcessDirectory(string inputDirectory, string outputDirectory)
        {
            Directory.CreateDirectory(outputDirectory);
            var directory = Directory.CreateDirectory(inputDirectory);
            foreach (var fileInfo in directory.GetFiles())
            {
                if (fileInfo.Attributes.HasFlag(FileAttributes.ReparsePoint))
                {
                    var targetPath = fileInfo.LinkTarget;
                    if (!string.IsNullOrEmpty(targetPath))
                    {
                        var tagetFile = Path.GetFullPath(targetPath, fileInfo.DirectoryName);
                        if (File.Exists(tagetFile))
                        {
                            ProcessFile(tagetFile, Path.Combine(outputDirectory, fileInfo.Name));
                        }
                    }
                }
                else
                {
                    ProcessFile(fileInfo.FullName, Path.Combine(outputDirectory, fileInfo.Name));
                }
            }
        }


        private static void ProcessFile(string source, string target)
        {
            File.Move(source, target);
        }


        private static DirectoryInfo GetSnapshot(string source)
        {
            var sourceDirectory = new DirectoryInfo(Path.Combine(source, "snapshots"));
            if (!sourceDirectory.Exists)
                return null;

            var snapshots = sourceDirectory.GetDirectories();
            if (snapshots == null || snapshots.Length == 0)
                return null;

            return snapshots.OrderByDescending(d => d.LastWriteTime).FirstOrDefault();
        }

        private record AmuseMigration(string Type, string OldName, string NewName);
        private record RenameMigration(string Type, string OldName, string NewName);
        private record HuggingFaceMigration(string Type, string OldName, string NewName);
    }


    public interface IMigrationService
    {
        Task RunMigrationsAsync();
        Task RunAutoMigrationsAsync();
    }
}
