/* gain_lock.c — see gain_lock.h. */
#include "sdkconfig.h"

#if CONFIG_CSI_GAIN_LOCK

#include <stdbool.h>
#include <stdint.h>
#include "esp_log.h"
#include "esp_wifi.h"
#include "esp_csi_gain_ctrl.h"
#include "gain_lock.h"

static const char *TAG = "gain_lock";

static bool s_locked = false;
static uint32_t s_samples = 0;
static uint32_t s_since_log = 0;
static uint8_t s_locked_agc = 0;
static int8_t s_locked_fft = 0;

void gain_lock_on_packet(const wifi_pkt_rx_ctrl_t *rx_ctrl)
{
    uint8_t agc_gain = 0;
    int8_t fft_gain = 0;
    esp_csi_gain_ctrl_get_rx_gain((wifi_pkt_rx_ctrl_t *)rx_ctrl, &agc_gain, &fft_gain);

    if (s_locked) {
        /* Periodic verification the lock is holding (gains should be constant). */
        if ((++s_since_log % 500) == 0) {
            if (agc_gain != s_locked_agc || fft_gain != s_locked_fft) {
                ESP_LOGW(TAG, "gain moved despite lock: agc=%u (locked %u) fft=%d (locked %d)",
                         agc_gain, s_locked_agc, fft_gain, s_locked_fft);
            } else {
                ESP_LOGI(TAG, "lock holding: agc=%u fft=%d", agc_gain, fft_gain);
            }
        }
        return;
    }

    if (s_samples < CONFIG_CSI_GAIN_LOCK_SAMPLES) {
        esp_csi_gain_ctrl_record_rx_gain(agc_gain, fft_gain);
        s_samples++;
        return;
    }

    /* Baseline window full — lock both gains to the recorded baseline. */
    uint8_t agc_baseline = 0;
    int8_t fft_baseline = 0;
    esp_csi_gain_ctrl_get_rx_gain_baseline(&agc_baseline, &fft_baseline);
    esp_csi_gain_ctrl_set_rx_force_gain(agc_baseline, fft_baseline);
    s_locked = true;
    s_locked_agc = agc_baseline;
    s_locked_fft = fft_baseline;
    ESP_LOGI(TAG, "GAIN LOCKED after %lu samples: agc=%u fft=%d",
             (unsigned long)s_samples, agc_baseline, fft_baseline);
}

bool gain_lock_is_locked(void)
{
    return s_locked;
}

#endif /* CONFIG_CSI_GAIN_LOCK */
