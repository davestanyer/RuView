/* gain_lock.h — AGC/FFT gain lock for CSI amplitude stability (drift experiment).
 *
 * ESP32 CSI amplitude is rescaled per-packet by automatic gain control; the
 * gain value is not part of the public rx_ctrl struct, so absolute amplitude
 * statistics drift arbitrarily across packets/sessions (see
 * docs/handover/2026-07-04-weekend/THEORY.md §8a). This module samples the
 * per-packet AGC/FFT gains via Espressif's esp_csi_gain_ctrl component for
 * the first CONFIG_CSI_GAIN_LOCK_SAMPLES packets, then locks both gains to
 * their baseline so subsequent CSI amplitudes share one scale.
 *
 * Enabled per-build via CONFIG_CSI_GAIN_LOCK (default off). Deploy on ONE
 * node first and compare empty-room amplitude stability against an unlocked
 * node before fleet rollout.
 */
#pragma once

#include "esp_wifi.h"

#ifdef __cplusplus
extern "C" {
#endif

#if CONFIG_CSI_GAIN_LOCK
/* Call once per accepted CSI packet, before serialization. Samples gains
 * until the baseline window fills, then forces AGC+FFT gain once. */
void gain_lock_on_packet(const wifi_pkt_rx_ctrl_t *rx_ctrl);

/* True once the gain has been forced. */
bool gain_lock_is_locked(void);
#else
static inline void gain_lock_on_packet(const wifi_pkt_rx_ctrl_t *rx_ctrl) { (void)rx_ctrl; }
static inline bool gain_lock_is_locked(void) { return false; }
#endif

#ifdef __cplusplus
}
#endif
