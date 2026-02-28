#[derive(Debug, Clone, Copy, Default)]
pub struct SystemSnapshot {
    pub cpu_percent: f64,
    pub memory_rss_mib: u64,
}

pub struct SystemSampler {
    #[cfg(target_os = "linux")]
    last_proc_jiffies: u64,
    #[cfg(target_os = "linux")]
    last_total_jiffies: u64,

    #[cfg(target_os = "windows")]
    last_proc_time_100ns: u64,
    #[cfg(target_os = "windows")]
    last_sample_instant: std::time::Instant,
    #[cfg(target_os = "windows")]
    logical_cpus: u64,
}

impl SystemSampler {
    pub fn new() -> Self {
        #[cfg(target_os = "linux")]
        {
            let proc_jiffies = read_process_jiffies().unwrap_or(0);
            let total_jiffies = read_total_jiffies().unwrap_or(0);
            return Self {
                last_proc_jiffies: proc_jiffies,
                last_total_jiffies: total_jiffies,
            };
        }

        #[cfg(target_os = "windows")]
        {
            let proc_time_100ns = read_process_time_100ns().unwrap_or(0);
            let logical_cpus = std::thread::available_parallelism()
                .map(|value| value.get() as u64)
                .unwrap_or(1)
                .max(1);
            return Self {
                last_proc_time_100ns: proc_time_100ns,
                last_sample_instant: std::time::Instant::now(),
                logical_cpus,
            };
        }

        #[cfg(not(any(target_os = "linux", target_os = "windows")))]
        {
            Self {}
        }
    }

    pub fn sample(&mut self) -> SystemSnapshot {
        #[cfg(target_os = "linux")]
        {
            let proc_jiffies = read_process_jiffies().unwrap_or(self.last_proc_jiffies);
            let total_jiffies = read_total_jiffies().unwrap_or(self.last_total_jiffies);

            let delta_proc = proc_jiffies.saturating_sub(self.last_proc_jiffies);
            let delta_total = total_jiffies.saturating_sub(self.last_total_jiffies).max(1);

            self.last_proc_jiffies = proc_jiffies;
            self.last_total_jiffies = total_jiffies;

            let cpu_percent = (delta_proc as f64 / delta_total as f64) * 100.0;
            let memory_rss_mib = read_memory_rss_mib().unwrap_or(0);
            return SystemSnapshot {
                cpu_percent,
                memory_rss_mib,
            };
        }

        #[cfg(target_os = "windows")]
        {
            let proc_time_100ns = read_process_time_100ns().unwrap_or(self.last_proc_time_100ns);
            let delta_proc = proc_time_100ns.saturating_sub(self.last_proc_time_100ns);

            let now = std::time::Instant::now();
            let delta_wall_100ns = now
                .duration_since(self.last_sample_instant)
                .as_nanos()
                .saturating_div(100)
                .max(1) as u64;
            let normalized_capacity_100ns =
                delta_wall_100ns.saturating_mul(self.logical_cpus).max(1);

            self.last_proc_time_100ns = proc_time_100ns;
            self.last_sample_instant = now;

            let cpu_percent =
                ((delta_proc as f64 / normalized_capacity_100ns as f64) * 100.0).clamp(0.0, 100.0);
            let memory_rss_mib = read_memory_rss_mib_windows().unwrap_or(0);
            return SystemSnapshot {
                cpu_percent,
                memory_rss_mib,
            };
        }

        #[cfg(not(any(target_os = "linux", target_os = "windows")))]
        {
            SystemSnapshot::default()
        }
    }
}

#[cfg(target_os = "linux")]
fn read_process_jiffies() -> Option<u64> {
    let stat = std::fs::read_to_string("/proc/self/stat").ok()?;

    let right_paren = stat.rfind(')')?;
    let tail = stat.get(right_paren + 2..)?;
    let fields: Vec<&str> = tail.split_whitespace().collect();

    // utime/stime are field 14/15 in /proc/self/stat, index 11/12 in this tail slice.
    let utime = fields.get(11)?.parse::<u64>().ok()?;
    let stime = fields.get(12)?.parse::<u64>().ok()?;

    Some(utime.saturating_add(stime))
}

#[cfg(target_os = "linux")]
fn read_total_jiffies() -> Option<u64> {
    let stat = std::fs::read_to_string("/proc/stat").ok()?;
    let first = stat.lines().next()?;
    let mut parts = first.split_whitespace();

    if parts.next()? != "cpu" {
        return None;
    }

    let mut total: u64 = 0;
    for value in parts {
        total = total.saturating_add(value.parse::<u64>().ok()?);
    }

    Some(total)
}

#[cfg(target_os = "linux")]
fn read_memory_rss_mib() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;

    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("VmRSS:") {
            let kib = rest
                .split_whitespace()
                .next()
                .and_then(|value| value.parse::<u64>().ok())?;
            return Some(kib / 1024);
        }
    }

    None
}

#[cfg(target_os = "windows")]
fn filetime_to_u64(file_time: windows_sys::Win32::Foundation::FILETIME) -> u64 {
    ((file_time.dwHighDateTime as u64) << 32) | (file_time.dwLowDateTime as u64)
}

#[cfg(target_os = "windows")]
fn read_process_time_100ns() -> Option<u64> {
    use std::mem::zeroed;
    use windows_sys::Win32::Foundation::FILETIME;
    use windows_sys::Win32::System::Threading::{GetCurrentProcess, GetProcessTimes};

    // SAFETY: We pass valid pointers to stack-allocated FILETIME outputs.
    unsafe {
        let process = GetCurrentProcess();
        let mut creation_time: FILETIME = zeroed();
        let mut exit_time: FILETIME = zeroed();
        let mut kernel_time: FILETIME = zeroed();
        let mut user_time: FILETIME = zeroed();
        if GetProcessTimes(
            process,
            &mut creation_time,
            &mut exit_time,
            &mut kernel_time,
            &mut user_time,
        ) == 0
        {
            return None;
        }
        Some(filetime_to_u64(kernel_time).saturating_add(filetime_to_u64(user_time)))
    }
}

#[cfg(target_os = "windows")]
fn read_memory_rss_mib_windows() -> Option<u64> {
    use std::mem::{size_of, zeroed};
    use windows_sys::Win32::System::ProcessStatus::{
        GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;

    // SAFETY: We pass a valid process handle and initialized PROCESS_MEMORY_COUNTERS buffer.
    unsafe {
        let process = GetCurrentProcess();
        let mut counters: PROCESS_MEMORY_COUNTERS = zeroed();
        counters.cb = size_of::<PROCESS_MEMORY_COUNTERS>() as u32;
        if GetProcessMemoryInfo(process, &mut counters, counters.cb) == 0 {
            return None;
        }
        Some((counters.WorkingSetSize as u64) / (1024 * 1024))
    }
}
