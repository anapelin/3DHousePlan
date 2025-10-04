"""
Generate HTML reports for reconstruction results
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
import logging
from datetime import datetime
import shutil

logger = logging.getLogger("colmap_pipeline")


class ReportGenerator:
    """Generate HTML reports for reconstruction results."""
    
    def __init__(self, workspace: Path):
        """
        Initialize report generator.
        
        Args:
            workspace: Workspace directory
        """
        self.workspace = workspace
        self.report_dir = workspace / "report"
        self.report_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_report(
        self,
        stats: Dict,
        config: Dict,
        timings: Dict[str, float]
    ) -> Path:
        """
        Generate HTML report.
        
        Args:
            stats: Reconstruction statistics
            config: Configuration used
            timings: Stage timings
            
        Returns:
            Path to generated report
        """
        logger.info("Generating HTML report...")
        
        report_path = self.report_dir / "report.html"
        
        html = self._generate_html(stats, config, timings)
        
        with open(report_path, "w") as f:
            f.write(html)
        
        # Save stats as JSON
        stats_path = self.report_dir / "stats.json"
        with open(stats_path, "w") as f:
            json.dump({
                "stats": stats,
                "timings": timings,
                "timestamp": datetime.now().isoformat()
            }, f, indent=2)
        
        logger.info(f"Report generated: {report_path}")
        return report_path
    
    def _generate_html(
        self,
        stats: Dict,
        config: Dict,
        timings: Dict[str, float]
    ) -> str:
        """Generate HTML content."""
        
        # Generate thumbnails section
        thumbnails_html = self._generate_thumbnails_section()
        
        # Generate stats section
        stats_html = self._generate_stats_section(stats)
        
        # Generate timings section
        timings_html = self._generate_timings_section(timings)
        
        # Generate config section
        config_html = self._generate_config_section(config)
        
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>COLMAP Reconstruction Report</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            line-height: 1.6;
            color: #333;
            background: #f5f5f5;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        
        header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px 0;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        
        h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        
        .subtitle {{
            opacity: 0.9;
            font-size: 1.1em;
        }}
        
        .section {{
            background: white;
            border-radius: 8px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        h2 {{
            color: #667eea;
            margin-bottom: 20px;
            font-size: 1.8em;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}
        
        .stat-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        .stat-label {{
            font-size: 0.9em;
            opacity: 0.9;
            margin-bottom: 5px;
        }}
        
        .stat-value {{
            font-size: 2em;
            font-weight: bold;
        }}
        
        .thumbnails {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }}
        
        .thumbnail {{
            position: relative;
            padding-bottom: 75%;
            background: #f0f0f0;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        .thumbnail img {{
            position: absolute;
            width: 100%;
            height: 100%;
            object-fit: cover;
        }}
        
        .thumbnail-label {{
            position: absolute;
            bottom: 0;
            left: 0;
            right: 0;
            background: rgba(0,0,0,0.7);
            color: white;
            padding: 5px 10px;
            font-size: 0.85em;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }}
        
        th, td {{
            text-align: left;
            padding: 12px;
            border-bottom: 1px solid #eee;
        }}
        
        th {{
            background: #f8f8f8;
            font-weight: 600;
            color: #667eea;
        }}
        
        tr:hover {{
            background: #f8f8f8;
        }}
        
        .badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.85em;
            font-weight: 500;
        }}
        
        .badge-success {{
            background: #d4edda;
            color: #155724;
        }}
        
        .badge-warning {{
            background: #fff3cd;
            color: #856404;
        }}
        
        .badge-info {{
            background: #d1ecf1;
            color: #0c5460;
        }}
        
        pre {{
            background: #f8f8f8;
            padding: 15px;
            border-radius: 6px;
            overflow-x: auto;
            font-size: 0.9em;
        }}
        
        footer {{
            text-align: center;
            padding: 20px;
            color: #666;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <header>
        <div class="container">
            <h1>🏗️ COLMAP Reconstruction Report</h1>
            <div class="subtitle">Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</div>
        </div>
    </header>
    
    <div class="container">
        {stats_html}
        {timings_html}
        {thumbnails_html}
        {config_html}
    </div>
    
    <footer>
        <p>Generated by COLMAP Pipeline v1.0.0</p>
    </footer>
</body>
</html>
        """
        
        return html
    
    def _generate_stats_section(self, stats: Dict) -> str:
        """Generate statistics section."""
        html = '<div class="section">\n'
        html += '<h2>Reconstruction Statistics</h2>\n'
        html += '<div class="stats-grid">\n'
        
        # Extract key stats
        stat_items = [
            ("Input Images", stats.get("num_images", "N/A")),
            ("Registered Images", stats.get("num_registered", "N/A")),
            ("3D Points (Sparse)", stats.get("num_sparse_points", "N/A")),
            ("3D Points (Dense)", stats.get("num_dense_points", "N/A")),
            ("Mesh Vertices", stats.get("num_mesh_vertices", "N/A")),
            ("Mesh Faces", stats.get("num_mesh_faces", "N/A")),
        ]
        
        for label, value in stat_items:
            html += f'''
            <div class="stat-card">
                <div class="stat-label">{label}</div>
                <div class="stat-value">{value:,}</div>
            </div>
            '''
        
        html += '</div>\n</div>\n'
        return html
    
    def _generate_timings_section(self, timings: Dict[str, float]) -> str:
        """Generate timings section."""
        html = '<div class="section">\n'
        html += '<h2>Stage Timings</h2>\n'
        html += '<table>\n'
        html += '<tr><th>Stage</th><th>Duration</th><th>Status</th></tr>\n'
        
        total_time = sum(timings.values())
        
        for stage, duration in timings.items():
            duration_str = self._format_duration(duration)
            percentage = (duration / total_time * 100) if total_time > 0 else 0
            
            html += f'''
            <tr>
                <td>{stage.replace("_", " ").title()}</td>
                <td>{duration_str} ({percentage:.1f}%)</td>
                <td><span class="badge badge-success">✓ Completed</span></td>
            </tr>
            '''
        
        html += f'''
        <tr style="font-weight: bold; background: #f0f0f0;">
            <td>Total</td>
            <td>{self._format_duration(total_time)}</td>
            <td><span class="badge badge-info">Complete</span></td>
        </tr>
        '''
        
        html += '</table>\n</div>\n'
        return html
    
    def _generate_thumbnails_section(self) -> str:
        """Generate thumbnails section."""
        images_dir = self.workspace / "images"
        
        if not images_dir.exists():
            return ""
        
        image_files = sorted(list(images_dir.glob("*.jpg"))[:12])  # First 12 images
        
        if not image_files:
            return ""
        
        html = '<div class="section">\n'
        html += '<h2>Sample Input Images</h2>\n'
        html += '<div class="thumbnails">\n'
        
        for i, img_path in enumerate(image_files):
            # Copy thumbnail to report directory
            thumb_dest = self.report_dir / f"thumb_{i:03d}.jpg"
            try:
                shutil.copy2(img_path, thumb_dest)
                rel_path = f"thumb_{i:03d}.jpg"
                
                html += f'''
                <div class="thumbnail">
                    <img src="{rel_path}" alt="{img_path.name}">
                    <div class="thumbnail-label">{img_path.name}</div>
                </div>
                '''
            except Exception as e:
                logger.warning(f"Failed to copy thumbnail: {e}")
        
        html += '</div>\n</div>\n'
        return html
    
    def _generate_config_section(self, config: Dict) -> str:
        """Generate configuration section."""
        html = '<div class="section">\n'
        html += '<h2>Configuration</h2>\n'
        
        # Extract key config items
        config_str = json.dumps(config, indent=2)
        html += f'<pre>{config_str}</pre>\n'
        
        html += '</div>\n'
        return html
    
    def _format_duration(self, seconds: float) -> str:
        """Format duration in seconds to human-readable string."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}m {secs}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"

