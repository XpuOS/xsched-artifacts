// Real-time speech recognition of input from a microphone
//
// A very quick-n-dirty implementation serving mainly as a proof of concept.
//
#include "common-sdl.h"
#include "common.h"
#include "whisper.h"

#include <cassert>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>
#include <fstream>
#include <chrono>

// Simulated microphone that reads from a WAV file
class wav_file_audio {
public:
    wav_file_audio(int len_ms) {
        m_len_ms = len_ms;
        m_running = false;
        m_sample_rate = WHISPER_SAMPLE_RATE;
        m_pos = 0;
    }

    ~wav_file_audio() {
        // Nothing to do here
    }

    bool init(const std::string& wav_path, int sample_rate) {
        m_sample_rate = sample_rate;

        // Read the entire WAV file
        if (!read_wav(wav_path, m_audio_data, m_audio_data_stereo, false)) {
            fprintf(stderr, "%s: failed to read WAV file '%s'!\n", __func__, wav_path.c_str());
            return false;
        }

        fprintf(stderr, "%s: loaded WAV file with %zu samples\n", __func__, m_audio_data.size());
        return true;
    }

    bool resume() {
        if (m_running) {
            fprintf(stderr, "%s: already running!\n", __func__);
            return false;
        }

        m_running = true;
        return true;
    }

    bool pause() {
        if (!m_running) {
            fprintf(stderr, "%s: already paused!\n", __func__);
            return false;
        }

        m_running = false;
        return true;
    }

    bool clear() {
        if (!m_running) {
            fprintf(stderr, "%s: not running!\n", __func__);
            return false;
        }

        // Do NOT reset position - we want to continue from where we are
        // Just acknowledge the audio has been processed
        return true;
    }

    void get(int ms, std::vector<float>& result) {
        if (!m_running) {
            fprintf(stderr, "%s: not running!\n", __func__);
            return;
        }

        result.clear();

        if (ms <= 0) {
            ms = m_len_ms;
        }

        // Calculate how many samples to read for requested milliseconds
        size_t n_samples = (m_sample_rate * ms) / 1000;
        
        // Check if we have enough samples left
        if (m_pos + n_samples > m_audio_data.size()) {
            // If we're at the end of the file, wrap around to the beginning
            // We want to return exactly n_samples if possible
            
            if (m_pos >= m_audio_data.size()) {
                // We're exactly at the end or beyond, reset to beginning
                m_pos = 0;
                fprintf(stderr, "%s: reached end of WAV file, restarting from beginning\n", __func__);
            }
            
            // Calculate how many samples we can get from current position to end
            size_t samples_from_current = m_audio_data.size() - m_pos;
            
            // Fill the first part from current position to end
            result.assign(m_audio_data.begin() + m_pos, m_audio_data.end());
            
            // Calculate how many more samples we need from the beginning
            size_t samples_from_beginning = n_samples - samples_from_current;
            
            // If we need more samples than the file has, cap it
            if (samples_from_beginning > m_audio_data.size()) {
                samples_from_beginning = m_audio_data.size();
            }
            
            // Add samples from the beginning of the file
            result.insert(result.end(), m_audio_data.begin(), m_audio_data.begin() + samples_from_beginning);
            
            // Update position for next read
            m_pos = samples_from_beginning;
        } else {
            // Standard case: just read the chunk
            result.assign(m_audio_data.begin() + m_pos, m_audio_data.begin() + m_pos + n_samples);
            m_pos += n_samples;
        }

        // Simulate real-time delay (uncomment if you want to simulate real time)
        // std::this_thread::sleep_for(std::chrono::milliseconds(ms));
    }

    void reset() {
        m_pos = 0;
        fprintf(stderr, "%s: manually reset to beginning of WAV file\n", __func__);
    }

private:
    int m_len_ms;
    int m_sample_rate;
    bool m_running;
    std::vector<float> m_audio_data;
    std::vector<std::vector<float>> m_audio_data_stereo;
    size_t m_pos;
};

// command-line parameters
struct whisper_params {
    int32_t n_threads  = std::min(4, (int32_t) std::thread::hardware_concurrency());
    int32_t step_ms    = 3000;
    int32_t length_ms  = 10000;
    int32_t keep_ms    = 200;
    int32_t capture_id = -1;
    int32_t max_tokens = 32;
    int32_t audio_ctx  = 0;

    float vad_thold    = 0.6f;
    float freq_thold   = 100.0f;

    bool translate     = false;
    bool no_fallback   = false;
    bool print_special = false;
    bool no_context    = true;
    bool no_timestamps = false;
    bool tinydiarize   = false;
    bool save_audio    = false; // save audio to wav file
    bool use_gpu       = true;
    bool flash_attn    = false;
    bool use_wav_file  = false; // use wav file instead of microphone

    std::string language  = "en";
    std::string model     = "models/ggml-base.en.bin";
    std::string fname_out;
    std::string wav_file; // path to wav file

    std::string openvino_encode_device = "CPU";
};

void whisper_print_usage(int argc, char ** argv, const whisper_params & params);

static bool whisper_params_parse(int argc, char ** argv, whisper_params & params) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            whisper_print_usage(argc, argv, params);
            exit(0);
        }
        else if (arg == "-t"    || arg == "--threads")       { params.n_threads     = std::stoi(argv[++i]); }
        else if (                  arg == "--step")          { params.step_ms       = std::stoi(argv[++i]); }
        else if (                  arg == "--length")        { params.length_ms     = std::stoi(argv[++i]); }
        else if (                  arg == "--keep")          { params.keep_ms       = std::stoi(argv[++i]); }
        else if (arg == "-c"    || arg == "--capture")       { params.capture_id    = std::stoi(argv[++i]); }
        else if (arg == "-mt"   || arg == "--max-tokens")    { params.max_tokens    = std::stoi(argv[++i]); }
        else if (arg == "-ac"   || arg == "--audio-ctx")     { params.audio_ctx     = std::stoi(argv[++i]); }
        else if (arg == "-vth"  || arg == "--vad-thold")     { params.vad_thold     = std::stof(argv[++i]); }
        else if (arg == "-fth"  || arg == "--freq-thold")    { params.freq_thold    = std::stof(argv[++i]); }
        else if (arg == "-tr"   || arg == "--translate")     { params.translate     = true; }
        else if (arg == "-nf"   || arg == "--no-fallback")   { params.no_fallback   = true; }
        else if (arg == "-ps"   || arg == "--print-special") { params.print_special = true; }
        else if (arg == "-kc"   || arg == "--keep-context")  { params.no_context    = false; }
        else if (arg == "-l"    || arg == "--language")      { params.language      = argv[++i]; }
        else if (arg == "-m"    || arg == "--model")         { params.model         = argv[++i]; }
        else if (arg == "-f"    || arg == "--file")          { params.fname_out     = argv[++i]; }
        else if (arg == "-tdrz" || arg == "--tinydiarize")   { params.tinydiarize   = true; }
        else if (arg == "-sa"   || arg == "--save-audio")    { params.save_audio    = true; }
        else if (arg == "-ng"   || arg == "--no-gpu")        { params.use_gpu       = false; }
        else if (arg == "-fa"   || arg == "--flash-attn")    { params.flash_attn    = true; }
        else if (arg == "-oved" || arg == "--ov-e-device")   { params.openvino_encode_device = argv[++i]; }
        else if (arg == "-wf"   || arg == "--wav-file")      { params.use_wav_file  = true; params.wav_file = argv[++i]; }
        else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            whisper_print_usage(argc, argv, params);
            exit(0);
        }
    }

    return true;
}

void whisper_print_usage(int /*argc*/, char ** argv, const whisper_params & params) {
    fprintf(stderr, "\n");
    fprintf(stderr, "usage: %s [options]\n", argv[0]);
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h,       --help          [default] show this help message and exit\n");
    fprintf(stderr, "  -t N,     --threads N     [%-7d] number of threads to use during computation\n",    params.n_threads);
    fprintf(stderr, "            --step N        [%-7d] audio step size in milliseconds\n",                params.step_ms);
    fprintf(stderr, "            --length N      [%-7d] audio length in milliseconds\n",                   params.length_ms);
    fprintf(stderr, "            --keep N        [%-7d] audio to keep from previous step in ms\n",         params.keep_ms);
    fprintf(stderr, "  -c ID,    --capture ID    [%-7d] capture device ID\n",                              params.capture_id);
    fprintf(stderr, "  -mt N,    --max-tokens N  [%-7d] maximum number of tokens per audio chunk\n",       params.max_tokens);
    fprintf(stderr, "  -ac N,    --audio-ctx N   [%-7d] audio context size (0 - all)\n",                   params.audio_ctx);
    fprintf(stderr, "  -vth N,   --vad-thold N   [%-7.2f] voice activity detection threshold\n",           params.vad_thold);
    fprintf(stderr, "  -fth N,   --freq-thold N  [%-7.2f] high-pass frequency cutoff\n",                   params.freq_thold);
    fprintf(stderr, "  -tr,      --translate     [%-7s] translate from source language to english\n",      params.translate ? "true" : "false");
    fprintf(stderr, "  -nf,      --no-fallback   [%-7s] do not use temperature fallback while decoding\n", params.no_fallback ? "true" : "false");
    fprintf(stderr, "  -ps,      --print-special [%-7s] print special tokens\n",                           params.print_special ? "true" : "false");
    fprintf(stderr, "  -kc,      --keep-context  [%-7s] keep context between audio chunks\n",              params.no_context ? "false" : "true");
    fprintf(stderr, "  -l LANG,  --language LANG [%-7s] spoken language\n",                                params.language.c_str());
    fprintf(stderr, "  -m FNAME, --model FNAME   [%-7s] model path\n",                                     params.model.c_str());
    fprintf(stderr, "  -f FNAME, --file FNAME    [%-7s] text output file name\n",                          params.fname_out.c_str());
    fprintf(stderr, "  -tdrz,    --tinydiarize   [%-7s] enable tinydiarize (requires a tdrz model)\n",     params.tinydiarize ? "true" : "false");
    fprintf(stderr, "  -sa,      --save-audio    [%-7s] save the recorded audio to a file\n",              params.save_audio ? "true" : "false");
    fprintf(stderr, "  -ng,      --no-gpu        [%-7s] disable GPU inference\n",                          params.use_gpu ? "false" : "true");
    fprintf(stderr, "  -fa,      --flash-attn    [%-7s] flash attention during inference\n",               params.flash_attn ? "true" : "false");
    fprintf(stderr, "  -oved D,   --ov-e-device DNAME [%-7s] the OpenVINO device used for encode inference\n",  params.openvino_encode_device.c_str());
    fprintf(stderr, "  -wf FNAME, --wav-file FNAME [%-7s] use WAV file as input instead of microphone\n", params.use_wav_file ? params.wav_file.c_str() : "false");
    fprintf(stderr, "\n");
}

int main(int argc, char ** argv) {
    whisper_params params;

    if (whisper_params_parse(argc, argv, params) == false) {
        return 1;
    }

    params.keep_ms   = std::min(params.keep_ms,   params.step_ms);
    params.length_ms = std::max(params.length_ms, params.step_ms);

    const int n_samples_step = (1e-3*params.step_ms  )*WHISPER_SAMPLE_RATE;
    const int n_samples_len  = (1e-3*params.length_ms)*WHISPER_SAMPLE_RATE;
    const int n_samples_keep = (1e-3*params.keep_ms  )*WHISPER_SAMPLE_RATE;
    const int n_samples_30s  = (1e-3*30000.0         )*WHISPER_SAMPLE_RATE;

    const bool use_vad = n_samples_step <= 0; // sliding window mode uses VAD

    const int n_new_line = !use_vad ? std::max(1, params.length_ms / params.step_ms - 1) : 1; // number of steps to print new line

    params.no_timestamps  = !use_vad;
    params.no_context    |= use_vad;
    params.max_tokens     = 0;

    // Init audio capture or wav file reader
    wav_file_audio* wav_audio = nullptr;
    audio_async* audio = nullptr;

    if (params.use_wav_file) {
        // Initialize WAV file reader
        wav_audio = new wav_file_audio(params.length_ms);
        if (!wav_audio->init(params.wav_file, WHISPER_SAMPLE_RATE)) {
            fprintf(stderr, "%s: wav_audio.init() failed!\n", __func__);
            delete wav_audio;
            return 1;
        }
        wav_audio->resume();
    } else {
        // Initialize microphone capture
        audio = new audio_async(params.length_ms);
        if (!audio->init(params.capture_id, WHISPER_SAMPLE_RATE)) {
            fprintf(stderr, "%s: audio.init() failed!\n", __func__);
            delete audio;
            return 1;
        }
        audio->resume();
    }

    // whisper init
    if (params.language != "auto" && whisper_lang_id(params.language.c_str()) == -1){
        fprintf(stderr, "error: unknown language '%s'\n", params.language.c_str());
        whisper_print_usage(argc, argv, params);
        exit(0);
    }

    struct whisper_context_params cparams = whisper_context_default_params();

    cparams.use_gpu    = params.use_gpu;
    cparams.flash_attn = params.flash_attn;

    struct whisper_context * ctx = whisper_init_from_file_with_params(params.model.c_str(), cparams);

    std::vector<float> pcmf32    (n_samples_30s, 0.0f);
    std::vector<float> pcmf32_old;
    std::vector<float> pcmf32_new(n_samples_30s, 0.0f);

    std::vector<whisper_token> prompt_tokens;

    if (whisper_ctx_init_openvino_encoder(ctx, nullptr, params.openvino_encode_device.c_str(), nullptr)) {
        fprintf(stderr, "%s: failed to initialize OpenVINO encoder\n", __func__);
        if (params.use_wav_file) delete wav_audio;
        else delete audio;
        return 2;
    }

    // print some info about the processing
    {
        fprintf(stderr, "\n");
        if (!whisper_is_multilingual(ctx)) {
            if (params.language != "en" || params.translate) {
                params.language = "en";
                params.translate = false;
                fprintf(stderr, "%s: WARNING: model is not multilingual, ignoring language and translation options\n", __func__);
            }
        }
        fprintf(stderr, "%s: processing %d samples (step = %.1f sec / len = %.1f sec / keep = %.1f sec), %d threads, lang = %s, task = %s, timestamps = %d ...\n",
                __func__,
                n_samples_step,
                float(n_samples_step)/WHISPER_SAMPLE_RATE,
                float(n_samples_len )/WHISPER_SAMPLE_RATE,
                float(n_samples_keep)/WHISPER_SAMPLE_RATE,
                params.n_threads,
                params.language.c_str(),
                params.translate ? "translate" : "transcribe",
                params.no_timestamps ? 0 : 1);

        if (!use_vad) {
            fprintf(stderr, "%s: n_new_line = %d, no_context = %d\n", __func__, n_new_line, params.no_context);
        } else {
            fprintf(stderr, "%s: using VAD, will transcribe on speech activity\n", __func__);
        }

        fprintf(stderr, "\n");
    }

    int n_iter = 0;

    bool is_running = true;

    std::ofstream fout;
    if (params.fname_out.length() > 0) {
        fout.open(params.fname_out);
        if (!fout.is_open()) {
            fprintf(stderr, "%s: failed to open output file '%s'!\n", __func__, params.fname_out.c_str());
            return 1;
        }
    }

    wav_writer wavWriter;
    // save wav file
    if (params.save_audio) {
        // Get current date/time for filename
        time_t now = time(0);
        char buffer[80];
        strftime(buffer, sizeof(buffer), "%Y%m%d%H%M%S", localtime(&now));
        std::string filename = std::string(buffer) + ".wav";

        wavWriter.open(filename, WHISPER_SAMPLE_RATE, 16, 1);
    }
    
    if (params.use_wav_file) {
        printf("[Processing WAV file: %s in endless loop mode]\n", params.wav_file.c_str());
    } else {
        printf("[Start speaking]\n");
    }
    fflush(stdout);

    auto t_last  = std::chrono::high_resolution_clock::now();
    const auto t_start = t_last;
    uint64_t segment_id = 0;

    auto t_last_get_audio = std::chrono::high_resolution_clock::now();

    // main audio loop
    while (is_running) {
        if (params.save_audio) {
            wavWriter.write(pcmf32_new.data(), pcmf32_new.size());
        }
        // handle Ctrl + C
        is_running = sdl_poll_events();

        if (!is_running) {
            break;
        }

        // process new audio

        if (!use_vad) {
            while (true) {
                if (params.use_wav_file) {
                    wav_audio->get(params.step_ms, pcmf32_new);
                    std::this_thread::sleep_until(t_last_get_audio + std::chrono::milliseconds(params.step_ms));
                    t_last_get_audio = std::chrono::high_resolution_clock::now();
                } else {
                    audio->get(params.step_ms, pcmf32_new);
                }

                // For WAV file input, we never end - we've already modified
                // the wav_file_audio class to loop back to the beginning

                if ((int) pcmf32_new.size() > 2*n_samples_step) {
                    fprintf(stderr, "\n\n%s: WARNING: cannot process audio fast enough, dropping audio ...\n\n", __func__);
                    if (params.use_wav_file) {
                        wav_audio->clear();  // Now only acknowledges data processed
                    } else {
                        audio->clear();
                    }
                    continue;
                }

                if ((int) pcmf32_new.size() >= n_samples_step) {
                    // Only clear after we've successfully read audio
                    if (params.use_wav_file) {
                        wav_audio->clear();  // Now only acknowledges data processed
                    } else {
                        audio->clear();
                    }
                    break;
                }

                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }

            // Normal processing continues

            const int n_samples_new = pcmf32_new.size();

            // take up to params.length_ms audio from previous iteration
            const int n_samples_take = std::min((int) pcmf32_old.size(), std::max(0, n_samples_keep + n_samples_len - n_samples_new));

            //printf("processing: take = %d, new = %d, old = %d\n", n_samples_take, n_samples_new, (int) pcmf32_old.size());

            pcmf32.resize(n_samples_new + n_samples_take);

            for (int i = 0; i < n_samples_take; i++) {
                pcmf32[i] = pcmf32_old[pcmf32_old.size() - n_samples_take + i];
            }

            memcpy(pcmf32.data() + n_samples_take, pcmf32_new.data(), n_samples_new*sizeof(float));

            pcmf32_old = pcmf32;
        } else {
            const auto t_now  = std::chrono::high_resolution_clock::now();
            const auto t_diff = std::chrono::duration_cast<std::chrono::milliseconds>(t_now - t_last).count();

            if (t_diff < 2000) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                continue;
            }

            if (params.use_wav_file) {
                wav_audio->get(2000, pcmf32_new);
                // Add sleep to simulate real-time playback speed
                std::this_thread::sleep_until(t_last_get_audio + std::chrono::milliseconds(2000));
                t_last_get_audio = std::chrono::high_resolution_clock::now();
            } else {
                audio->get(2000, pcmf32_new);
            }

            // For WAV file, we never reach the end - the class will loop back to the beginning

            if (::vad_simple(pcmf32_new, WHISPER_SAMPLE_RATE, 1000, params.vad_thold, params.freq_thold, false)) {
                if (params.use_wav_file) {
                    wav_audio->get(params.length_ms, pcmf32);
                    // Add sleep to simulate real-time playback speed
                    std::this_thread::sleep_until(t_last_get_audio + std::chrono::milliseconds(params.length_ms));
                    t_last_get_audio = std::chrono::high_resolution_clock::now();
                } else {
                    audio->get(params.length_ms, pcmf32);
                }
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                continue;
            }

            t_last = t_now;
        }

        // run the inference
        {
            auto begin = std::chrono::system_clock::now();

            whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);

            wparams.print_progress   = false;
            wparams.print_special    = params.print_special;
            wparams.print_realtime   = false;
            wparams.print_timestamps = !params.no_timestamps;
            wparams.translate        = params.translate;
            wparams.single_segment   = !use_vad;
            wparams.max_tokens       = params.max_tokens;
            wparams.language         = params.language.c_str();
            wparams.n_threads        = params.n_threads;

            wparams.audio_ctx        = params.audio_ctx;

            wparams.tdrz_enable      = params.tinydiarize; // [TDRZ]

            // disable temperature fallback
            //wparams.temperature_inc  = -1.0f;
            wparams.temperature_inc  = params.no_fallback ? 0.0f : wparams.temperature_inc;

            wparams.prompt_tokens    = params.no_context ? nullptr : prompt_tokens.data();
            wparams.prompt_n_tokens  = params.no_context ? 0       : prompt_tokens.size();

            if (whisper_full(ctx, wparams, pcmf32.data(), pcmf32.size()) != 0) {
                fprintf(stderr, "%s: failed to process audio\n", argv[0]);
                return 6;
            }

            // print result;
            {
                if (!use_vad) {
                    printf("\33[2K\r");

                    // print long empty line to clear the previous line
                    printf("%s", std::string(100, ' ').c_str());

                    printf("\33[2K\r");
                } else {
                    const int64_t t1 = (t_last - t_start).count()/1000000;
                    const int64_t t0 = std::max(0.0, t1 - pcmf32.size()*1000.0/WHISPER_SAMPLE_RATE);

                    printf("\n");
                    printf("### Transcription %d START | t0 = %d ms | t1 = %d ms\n", n_iter, (int) t0, (int) t1);
                    printf("\n");
                }

                const int n_segments = whisper_full_n_segments(ctx);
                for (int i = 0; i < n_segments; ++i) {
                    const char * text = whisper_full_get_segment_text(ctx, i);

                    if (params.no_timestamps) {
                        printf("%s", text);
                        fflush(stdout);

                        if (params.fname_out.length() > 0) {
                            fout << text;
                        }
                    } else {
                        const int64_t t0 = whisper_full_get_segment_t0(ctx, i);
                        const int64_t t1 = whisper_full_get_segment_t1(ctx, i);

                        std::string output = "[" + to_timestamp(t0, false) + " --> " + to_timestamp(t1, false) + "]  " + text;

                        if (whisper_full_get_segment_speaker_turn_next(ctx, i)) {
                            output += " [SPEAKER_TURN]";
                        }

                        output += "\n";

                        printf("%s", output.c_str());
                        fflush(stdout);

                        if (params.fname_out.length() > 0) {
                            fout << output;
                        }
                    }
                }

                if (params.fname_out.length() > 0) {
                    fout << std::endl;
                }

                if (use_vad) {
                    printf("\n");
                    printf("### Transcription %d END\n", n_iter);
                }
            }

            ++n_iter;

            if (!use_vad && (n_iter % n_new_line) == 0) {
                printf("\n");

                // keep part of the audio for next iteration to try to mitigate word boundary issues
                pcmf32_old = std::vector<float>(pcmf32.end() - n_samples_keep, pcmf32.end());

                // Add tokens of the last full length segment as the prompt
                if (!params.no_context) {
                    prompt_tokens.clear();

                    const int n_segments = whisper_full_n_segments(ctx);
                    for (int i = 0; i < n_segments; ++i) {
                        const int token_count = whisper_full_n_tokens(ctx, i);
                        for (int j = 0; j < token_count; ++j) {
                            prompt_tokens.push_back(whisper_full_get_token_id(ctx, i, j));
                        }
                    }
                }
            }
            fflush(stdout);

            auto end = std::chrono::system_clock::now();
            auto t = std::chrono::system_clock::to_time_t(end);
            auto ms = std::chrono::duration_cast<std::chrono::microseconds>(end.time_since_epoch()).count() % 1000000;
            std::tm* now = std::localtime(&t);
            double us = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
            fprintf(stderr, "{\"time\": \"%02d:%02d:%02d.%05ld\", \"segment_id\": %lu, \"latency (ms)\": %.3f},\n",
                    now->tm_hour, now->tm_min, now->tm_sec, ms, segment_id++, us / 1000);
            fflush(stderr);
        }
    }

    if (params.use_wav_file) {
        delete wav_audio;
    } else {
        audio->pause();
        delete audio;
    }

    whisper_print_timings(ctx);
    whisper_free(ctx);

    return 0;
}
