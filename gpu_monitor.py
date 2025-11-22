import subprocess
import threading
import time
from typing import Optional, Tuple


class GPUMonitor:
    """
    GPU-Monitor zur Messung des Stromverbrauchs über nvidia-smi.
    Misst alle 0,1 Sekunden den Watt-Verbrauch und berechnet Durchschnittswerte.
    """
    
    def __init__(self):
        self.measurements = []
        self.start_time = None
        self.end_time = None
        self.monitoring = False
        self.monitor_thread = None
    
    def _get_gpu_power(self) -> Optional[float]:
        """
        Ruft nvidia-smi auf und gibt den aktuellen Stromverbrauch in Watt zurück.
        
        Returns:
            float: Stromverbrauch in Watt oder None bei Fehler
        """
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                timeout=1
            )
            
            if result.returncode == 0:
                # Nimmt die erste GPU (Index 0) falls mehrere vorhanden sind
                power_str = result.stdout.strip().split('\n')[0]
                return float(power_str)
            return None
        except (subprocess.TimeoutExpired, subprocess.SubprocessError, ValueError, IndexError):
            return None
    
    def _monitor_loop(self):
        """
        Überwacht kontinuierlich den GPU-Stromverbrauch alle 0,1 Sekunden.
        """
        while self.monitoring:
            power = self._get_gpu_power()
            if power is not None:
                self.measurements.append(power)
            time.sleep(0.1)
    
    def start(self):
        """
        Startet die Messung des GPU-Stromverbrauchs und die Stoppuhr.
        """
        if self.monitoring:
            print("Messung läuft bereits!")
            return
        
        # Reset der Daten
        self.measurements = []
        self.start_time = time.time()
        self.end_time = None
        self.monitoring = True
        
        # Starte Monitoring-Thread
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        print("GPU-Monitoring gestartet...")
    
    def stop(self) -> Tuple[float, float]:
        """
        Stoppt die Messung und gibt die Ergebnisse zurück.
        
        Returns:
            Tuple[float, float]: (durchschnittlicher Watt-Verbrauch, Messdauer in Sekunden)
        """
        if not self.monitoring:
            print("Keine aktive Messung!")
            return 0.0, 0.0
        
        # Stoppe Monitoring
        self.monitoring = False
        self.end_time = time.time()
        
        # Warte auf Thread-Beendigung
        if self.monitor_thread is not None:
            self.monitor_thread.join(timeout=1.0)
        
        # Berechne Ergebnisse
        duration = self.end_time - self.start_time
        
        if len(self.measurements) == 0:
            print("Keine Messungen erfasst!")
            return 0.0, duration
        
        mean_watt = sum(self.measurements) / len(self.measurements)
        
        print(f"GPU-Monitoring gestoppt.")
        print(f"Messdauer: {duration:.2f} Sekunden")
        print(f"Anzahl Messungen: {len(self.measurements)}")
        print(f"Durchschnittlicher Verbrauch: {mean_watt:.2f} W")
        
        return mean_watt, duration
    
    def get_stats(self) -> dict:
        """
        Gibt detaillierte Statistiken über die Messungen zurück.
        
        Returns:
            dict: Statistiken inkl. min, max, mean, duration, measurement_count
        """
        if not self.measurements:
            return {
                'mean_watt': 0.0,
                'min_watt': 0.0,
                'max_watt': 0.0,
                'duration': 0.0,
                'measurement_count': 0
            }
        
        duration = (self.end_time or time.time()) - (self.start_time or 0)
        
        return {
            'mean_watt': sum(self.measurements) / len(self.measurements),
            'min_watt': min(self.measurements),
            'max_watt': max(self.measurements),
            'duration': duration,
            'measurement_count': len(self.measurements)
        }


# Beispielverwendung
if __name__ == "__main__":
    monitor = GPUMonitor()
    
    # Starte Messung
    monitor.start()
    
    # Simuliere Arbeit (z.B. Training)
    print("Führe GPU-intensive Aufgabe aus...")
    time.sleep(5)  # Beispiel: 5 Sekunden
    
    # Stoppe Messung und erhalte Ergebnisse
    mean_watt, duration = monitor.stop()
    
    # Detaillierte Statistiken
    stats = monitor.get_stats()
    print(stats)
