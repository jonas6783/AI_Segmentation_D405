import open3d as o3d
import numpy as np

class AlignmentViewer:
    """
    Klasse zur Visualisierung des Ergebnisses.
    Inklusive Achsen-Beschriftung (X, Y, Z) und Bodengitter.
    """
    def __init__(self, aligned_scan, cad_geometry, window_name="Ergebnis Viewer"):
        self.window_name = window_name
        self.scan = aligned_scan
        
        # CAD als Drahtgitter
        if isinstance(cad_geometry, o3d.geometry.TriangleMesh):
            self.cad_wire = o3d.geometry.LineSet.create_from_triangle_mesh(cad_geometry)
        else:
            self.cad_wire = cad_geometry 
        
        self.cad_wire.paint_uniform_color([0.6, 0.6, 0.6]) # Grau

        # --- ORIENTIERUNGSHILFEN ---
        self.helpers = []
        
        # 1. Achsenkreuz mit Buchstaben
        self.helpers.extend(self.create_labeled_axes(size=0.1))
        
        # 2. Bodengitter (XZ-Ebene, da Y bei dir wahrscheinlich Oben ist)
        self.helpers.append(self.create_grid(size=0.2, step=0.01))

    def create_grid(self, size=0.2, step=0.01):
        """Erzeugt ein graues Gitter auf der XZ-Ebene (Y=0)."""
        lines = []
        points = []
        
        # Bereich von -size bis +size
        r = np.arange(-size, size + step, step)
        
        # Linien parallel zur X-Achse
        for z in r:
            points.append([-size, 0, z])
            points.append([size, 0, z])
            lines.append([len(points)-2, len(points)-1])
            
        # Linien parallel zur Z-Achse
        for x in r:
            points.append([x, 0, -size])
            points.append([x, 0, size])
            lines.append([len(points)-2, len(points)-1])
            
        grid = o3d.geometry.LineSet()
        grid.points = o3d.utility.Vector3dVector(points)
        grid.lines = o3d.utility.Vector2iVector(lines)
        grid.paint_uniform_color([0.8, 0.8, 0.8]) # Hellgrau
        return grid

    def create_labeled_axes(self, size=0.1):
        """Erzeugt bunte Pfeile UND Buchstaben X, Y, Z aus Linien."""
        geoms = []
        
        # Standard Achsen (Rot/Grün/Blau)
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
        geoms.append(axes)
        
        # Buchstaben zeichnen (Vektor-Grafik Stil)
        def make_letter(lines_rel, offset, color):
            pts = np.array(lines_rel) * (size * 0.1) + offset # Skalieren & Verschieben
            lines = [[i, i+1] for i in range(0, len(pts), 2)]
            ls = o3d.geometry.LineSet()
            ls.points = o3d.utility.Vector3dVector(pts)
            ls.lines = o3d.utility.Vector2iVector(lines)
            ls.paint_uniform_color(color)
            return ls

        # X (Rot) - Zwei gekreuzte Linien
        geoms.append(make_letter(
            [[0,0,0], [1,1,0], [0,1,0], [1,0,0]], # Form X
            np.array([size*1.1, 0, 0]),           # Position
            [1, 0, 0]                             # Farbe Rot
        ))
        
        # Y (Grün) - Ein Ypsilon
        geoms.append(make_letter(
            [[0,1,0], [0.5,0.5,0], [1,1,0], [0.5,0.5,0], [0.5,0.5,0], [0.5,0,0]], # Form Y
            np.array([0, size*1.1, 0]), 
            [0, 1, 0]
        ))
        
        # Z (Blau) - Ein Z
        geoms.append(make_letter(
            [[0,1,0], [1,1,0], [1,1,0], [0,0,0], [0,0,0], [1,0,0]], # Form Z
            np.array([0, 0, size*1.1]), 
            [0, 0, 1]
        ))
        
        return geoms

    def run(self):
        print("\n" + "="*40)
        print(f"   {self.window_name}")
        print("="*40)
        print(" ORIENTIERUNG:")
        print("  Rot   (X) = Buchstaben 'X'")
        print("  Grün  (Y) = Buchstaben 'Y' (Oben/Unten?)")
        print("  Blau  (Z) = Buchstaben 'Z'")
        print("  Gitter    = Boden (XZ-Ebene)")
        print("-" * 40)
        print(" [1] Nur CAD")
        print(" [2] Nur Scan")
        print(" [3] Overlay (Standard)")
        print(" [Q] Schließen")

        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(window_name=self.window_name, width=1280, height=800)

        # Alles hinzufügen
        vis.add_geometry(self.cad_wire)
        vis.add_geometry(self.scan)
        for h in self.helpers:
            vis.add_geometry(h)

        # Callbacks
        def show_cad(vis):
            vis.clear_geometries()
            vis.add_geometry(self.cad_wire)
            for h in self.helpers: vis.add_geometry(h)
            return False

        def show_scan(vis):
            vis.clear_geometries()
            vis.add_geometry(self.scan)
            for h in self.helpers: vis.add_geometry(h)
            return False

        def show_overlay(vis):
            vis.clear_geometries()
            vis.add_geometry(self.cad_wire)
            vis.add_geometry(self.scan)
            for h in self.helpers: vis.add_geometry(h)
            return False

        vis.register_key_callback(ord("1"), show_cad)
        vis.register_key_callback(ord("2"), show_scan)
        vis.register_key_callback(ord("3"), show_overlay)
        vis.register_key_callback(ord("Q"), lambda v: v.close())

        vis.run()
        vis.destroy_window()
