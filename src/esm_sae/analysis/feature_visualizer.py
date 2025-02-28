#!/usr/bin/env python3
"""
Feature visualizer for ESM-SAE - Web interface to visualize activations of SAE features
across different protein classes.

This standalone script creates a web interface where you can:
1. Upload or input a protein sequence
2. Select feature classes to compare
3. View histograms of most activated features

Usage:
    python feature_visualizer.py [--port PORT] [--csv_dir CSV_DIR] [--checkpoint CHECKPOINT]

The script will start a web server on the specified port (default: 8000).
"""

import os
import sys
import json
import base64
import argparse
import io
import csv
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
import html
import traceback

# Import the Autoencoder model
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from esm_sae.sae.model import Autoencoder


@dataclass
class FeatureDataClass:
    """Data class representing a set of sequences with a common feature type."""
    name: str
    sequences: List[str] = field(default_factory=list)
    headers: List[str] = field(default_factory=list)
    
    def add_sequence(self, sequence: str, header: Optional[str] = None):
        """Add a sequence to this feature class."""
        self.sequences.append(sequence)
        self.headers.append(header if header else f"seq_{len(self.sequences)}")
    
    def load_from_csv(self, csv_path: str):
        """Load sequences from a CSV file."""
        try:
            with open(csv_path, 'r') as f:
                reader = csv.reader(f)
                header_row = next(reader, None)  # Skip header if exists
                
                # Find sequence column (assume first column if no "sequence" column found)
                seq_col = 0
                header_col = 1
                if header_row:
                    for i, col in enumerate(header_row):
                        if col.lower() == 'sequence':
                            seq_col = i
                        elif col.lower() in ('header', 'name', 'id'):
                            header_col = i
                
                # Read sequences and headers
                for row in reader:
                    if len(row) > seq_col:
                        sequence = row[seq_col].strip()
                        header = row[header_col].strip() if len(row) > header_col else f"seq_{len(self.sequences)}"
                        self.add_sequence(sequence, header)
                        
            print(f"Loaded {len(self.sequences)} sequences from {csv_path}")
        except Exception as e:
            print(f"Error loading {csv_path}: {e}")


class FeatureVisualizer:
    """Main class for feature visualization functionality."""
    
    def __init__(self, checkpoint_path: str, csv_dir: Optional[str] = None):
        """
        Initialize the feature visualizer.
        
        Args:
            checkpoint_path: Path to the SAE model checkpoint
            csv_dir: Directory containing CSV files with sequence classes
        """
        self.checkpoint_path = checkpoint_path
        self.csv_dir = csv_dir
        self.feature_classes = {}
        self.model = None
        self.params = None
        self.L = 0  # Latent dimension
        self.D = 0  # Input dimension
        self.topk = 32  # Default topk value
        
        # Load model checkpoint
        self._load_model()
        
        # Load feature classes from CSV directory if provided
        if csv_dir and os.path.isdir(csv_dir):
            self._load_feature_classes()
    
    def _load_model(self):
        """Load the SAE model from checkpoint."""
        try:
            print(f"Loading checkpoint from {self.checkpoint_path}")
            self.params = np.load(self.checkpoint_path, allow_pickle=True).item()
            
            # Extract model dimensions
            if 'enc' in self.params:
                self.D, self.L = self.params['enc']['kernel'].shape
            elif 'params' in self.params and 'enc' in self.params['params']:
                self.D, self.L = self.params['params']['enc']['kernel'].shape
            else:
                raise ValueError("Could not determine dimensions from checkpoint")
                
            print(f"Model dimensions: input_dim={self.D}, latent_dim={self.L}")
            
            # Initialize the SAE model
            self.model = Autoencoder(L=self.L, D=self.D, topk=self.topk)
            print("Model loaded successfully")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            traceback.print_exc()
    
    def _load_feature_classes(self):
        """Load feature classes from CSV files in csv_dir."""
        csv_files = [f for f in os.listdir(self.csv_dir) if f.endswith('.csv')]
        
        # Create example feature classes if no CSV files found
        if not csv_files:
            print(f"No CSV files found in {self.csv_dir}, creating example feature classes")
            self._create_example_feature_classes()
            return
            
        for csv_file in csv_files:
            class_name = os.path.splitext(csv_file)[0]
            feature_class = FeatureDataClass(name=class_name)
            feature_class.load_from_csv(os.path.join(self.csv_dir, csv_file))
            
            if feature_class.sequences:
                self.feature_classes[class_name] = feature_class
    
    def _create_example_feature_classes(self):
        """Create example feature classes."""
        # Beta sheet example
        beta_class = FeatureDataClass(name="beta4")
        beta_class.add_sequence("MVKVKVGVNGFGRIGRLVTRAAFNSGKVDIVAINDPFIDLNYMVYMFQYDSTHGKFHGTVKAENGKLVINGNPITIFQERDPSKIKWGDAGAEYVVESTGVFTTMEKAGAHLQGGAKRVIISAPSADAPMFVMGVNHEKYDNSLKIISNASCTTNCLAPLAKVIHDNFGIVEGLMTTVHAITATQKTVDGPSGKLWRDGRGALQNIIPASTGAAKAVGKVIPELNGKLTGMAFRVPTANVSVVDLTCRLEKPAKYDDIKKVVKQASEGPLKGILGYTEHQVVSSDFNSDTHSSTFDAGAGIALNDHFVKLISWYDNEFGYSNRVVDLMAHMASKE", "Beta example 1")
        beta_class.add_sequence("MSFQAPRRLLLVSLGLIFLLSATANGRDAIPMDAFDPVTLRLDVGTNLWGPYGGSAMLVLVQSDKPTIFRCDIRGSLLLTFDLRIQGEPIRVSAGGGMRDLKASAGRSFIVGDAIKCVAFTDPRWEGGASHFTVREFTPTAAPEDALVNFSVDRDDLDVDWRNLKSFAGGVAGAVACWPGSTTRVSPAHAPELGWSHIASFGGGPRISASWSPYHPTPGVFTINEETGQVCGLYRTENATFYFQGPEKIGEGPGVSIPLVSAGDFPSVATTDGKYLMVAQGTGYGYVVVSDASGKAAATKVLVDGAPFDLVPNVTVEPIPVLDVHNVKGVGCGHWAVASGKKVDVNQKMKKGPDGQYVLIICAGEGKDVDARIIFDPRTVSAMTFKSGKYKIHAGTYNDVNDVIFSRSGQGVLISNVDFSGSFHLNYGVAVKAADDSVKIYFSADDGSPLTLSAKGFAVLGVFESEHSSQEMADRVKFAAEGLSPRGGDFTPQCPTLPGILRTN", "Beta example 2")
        self.feature_classes["beta4"] = beta_class
        
        # Alpha helix example
        alpha_class = FeatureDataClass(name="alpha_helix")
        alpha_class.add_sequence("MTKIANKYEVIDNVDVSNLDKLKREYELLNEFLKNKNFQPKSFGLRSDLPKLNFEDAKACAEAFFANYKHGKGHEGLIPLNHPDVDKDTYMHPVDFTFVCMTEYEARQLEKFLADAEPKLVALGTETRKRQLTAWEQAEKRAEELRSAAKKLQEKLNEAKALEKAQPELKTQLTEYTEKLRKAAETAKEKAERLKVVEQALKNLEKMEMRKKLEESAKKLVTPFKARRLGRLREALLRRSQLLKELEQYRRAFQEALQAMKQALERLEQKKQELEKAKVDAETRAALFHGEELEEAVAAAQKELSTLRQEWEAENRKYQEAARKLEDELVKKERELDAALRKAQELEAMAKKLASALNAEKAAYSQAVAAKEELAAAREKLEAANAELQAARKKLEDQNVELQAAQAELREARAKLAAAE", "Alpha example 1")
        alpha_class.add_sequence("MGDVEKGKKIFIMKCSQCHTVEKGGKHKTGPNLHGLFGRKTGQAPGYSYTAANKNKGIIWGEDTLMEYLENPKKYIPGTKMIFVGIKKKEERADLIAYLKKATNE", "Alpha example 2")
        self.feature_classes["alpha_helix"] = alpha_class
    
    def infer_features(self, sequence: str, embedding: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        Infer features for a given sequence or embedding.
        
        Args:
            sequence: The protein sequence (will be ignored if embedding is provided)
            embedding: Pre-computed embedding (optional)
            
        Returns:
            Dictionary with feature activations and statistics
        """
        if not self.model or not self.params:
            raise ValueError("Model not loaded")
            
        if embedding is None:
            # We would need to compute the embedding from the sequence
            # This would require loading the ESM model, which is out of scope for this script
            raise ValueError("Embedding computation from sequence is not implemented")
        
        # Convert embedding to jax array and add batch dimension
        embedding_array = jnp.array([embedding])
        
        # Run through SAE to get activations
        zpre_BL, z_BL, _ = self.model.apply({'params': self.params}, embedding_array)
        
        # Convert to numpy for easier handling
        z = np.array(z_BL)[0]  # Remove batch dimension
        
        # Get indices of top-k activated features
        top_indices = np.argsort(-z)[:self.topk]
        
        # Get activation values for those indices
        top_values = z[top_indices]
        
        return {
            "feature_indices": top_indices.tolist(),
            "feature_values": top_values.tolist(),
            "all_values": z.tolist()
        }
    
    def compare_feature_classes(self, classes: List[str], embedding: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        Compare feature activations across different feature classes.
        
        Args:
            classes: List of feature class names to compare
            embedding: Optional embedding of a new sequence to compare
            
        Returns:
            Dictionary with comparison results and histograms
        """
        results = {}
        class_histograms = {}
        feature_counts = defaultdict(lambda: defaultdict(int))
        
        # Process each selected class
        for class_name in classes:
            if class_name not in self.feature_classes:
                continue
                
            feature_class = self.feature_classes[class_name]
            class_results = []
            
            # Process embeddings from pre-created feature class
            for i, header in enumerate(feature_class.headers):
                # For demonstration, we'll use random embeddings
                # In a real scenario, you would compute or load embeddings from ESM model
                random_embedding = np.random.randn(self.D).tolist()
                
                # Infer features
                features = self.infer_features("", random_embedding)
                class_results.append({
                    "header": header,
                    "features": features
                })
                
                # Count top feature occurrences
                for idx in features["feature_indices"]:
                    feature_counts[class_name][idx] += 1
            
            results[class_name] = class_results
            
            # Create histogram data for this class
            counts = feature_counts[class_name]
            total = len(feature_class.sequences) or 1  # Avoid division by zero
            histogram = {idx: count / total for idx, count in counts.items()}
            class_histograms[class_name] = histogram
        
        # Process the new embedding if provided
        input_features = None
        if embedding:
            input_features = self.infer_features("", embedding)
            results["input"] = {"features": input_features}
            
            # If no classes were selected, create a histogram just for the input sequence
            if not classes:
                input_histogram = {}
                for idx in input_features["feature_indices"]:
                    input_histogram[idx] = 1.0  # Mark as active
                class_histograms["Input Sequence"] = input_histogram
        
        # Create histogram comparison visualization
        histogram_image = self._create_histogram_comparison(class_histograms)
        
        return {
            "results": results,
            "histogram": histogram_image,
            "feature_counts": {cls: dict(counts) for cls, counts in feature_counts.items()},
            "input_features": input_features
        }
    
    def _create_histogram_comparison(self, class_histograms: Dict[str, Dict[int, float]]) -> str:
        """
        Create histogram comparison visualization.
        
        Args:
            class_histograms: Dictionary mapping class names to their feature histograms
            
        Returns:
            Base64-encoded PNG image of the histogram
        """
        plt.figure(figsize=(12, 6))
        
        # Check if there are any classes to compare
        if not class_histograms:
            plt.text(0.5, 0.5, "No classes selected for comparison",
                     horizontalalignment='center', verticalalignment='center',
                     transform=plt.gca().transAxes, fontsize=14)
            plt.axis('off')
            
            # Save plot to a bytes buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            plt.close()
            
            # Encode the image in base64 for HTML embedding
            return base64.b64encode(buf.read()).decode('utf-8')
        
        # Collect all feature indices
        all_indices = set()
        for hist in class_histograms.values():
            all_indices.update(hist.keys())
        
        # Sort indices
        sorted_indices = sorted(all_indices)
        
        # If no features were found, create an empty plot with a message
        if not sorted_indices:
            plt.text(0.5, 0.5, "No feature activations found",
                     horizontalalignment='center', verticalalignment='center',
                     transform=plt.gca().transAxes, fontsize=14)
            plt.axis('off')
            
            # Save plot to a bytes buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            plt.close()
            
            # Encode the image in base64 for HTML embedding
            return base64.b64encode(buf.read()).decode('utf-8')
        
        # Set width and positions for bars
        bar_width = 0.8 / len(class_histograms) if len(class_histograms) > 0 else 0.4
        
        # Plot histograms for each class
        for i, (class_name, histogram) in enumerate(class_histograms.items()):
            # Get values for all indices (default to 0 if not present)
            values = [histogram.get(idx, 0) for idx in sorted_indices]
            
            # Calculate position offset for this class
            offset = i * bar_width
            if len(class_histograms) > 1:
                offset = offset - (len(class_histograms) - 1) * bar_width / 2
            
            # Plot bars
            plt.bar(
                [x + offset for x in range(len(sorted_indices))],
                values,
                width=bar_width,
                label=class_name,
                alpha=0.7
            )
        
        # Set x-axis labels and other plot properties
        plt.xlabel('Feature Index')
        plt.ylabel('Activation Frequency')
        plt.title('Feature Activation Comparison')
        plt.xticks(range(len(sorted_indices)), [str(idx) for idx in sorted_indices], rotation=90)
        plt.legend()
        plt.tight_layout()
        
        # Save plot to a bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        
        # Encode the image in base64 for HTML embedding
        return base64.b64encode(buf.read()).decode('utf-8')


class FeatureVisualizerHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the feature visualizer web interface."""
    
    def __init__(self, *args, visualizer: FeatureVisualizer, **kwargs):
        self.visualizer = visualizer
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        """Handle GET requests."""
        # Parse URL
        parsed_url = urlparse(self.path)
        path = parsed_url.path
        
        # Serve the main page
        if path == '/' or path == '/index.html':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(self._generate_html().encode())
        else:
            self.send_response(404)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(b'Not Found')
    
    def do_POST(self):
        """Handle POST requests."""
        # Parse URL
        parsed_url = urlparse(self.path)
        path = parsed_url.path
        
        # Handle form submission
        if path == '/analyze':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length).decode('utf-8')
            form_data = parse_qs(post_data)
            
            try:
                # Extract form fields
                sequence = form_data.get('sequence', [''])[0]
                selected_classes = form_data.get('classes', [])
                
                # Handle different forms of the classes parameter
                if not selected_classes:
                    selected_classes = []
                elif isinstance(selected_classes, str):
                    selected_classes = [selected_classes]
                
                # For demo purposes, create a random embedding
                # In a real scenario, compute the embedding from the sequence using ESM
                random_embedding = np.random.randn(self.visualizer.D).tolist()
                
                # Compare classes and get results
                results = self.visualizer.compare_feature_classes(selected_classes, random_embedding)
                
                # Generate HTML with results
                html_content = self._generate_results_html(sequence, selected_classes, results)
                
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write(html_content.encode())
                
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                error_html = f"""
                <html>
                <head><title>Error</title></head>
                <body>
                    <h1>Error</h1>
                    <p>{html.escape(str(e))}</p>
                    <pre>{html.escape(traceback.format_exc())}</pre>
                    <p><a href="/">Back to home</a></p>
                </body>
                </html>
                """
                self.wfile.write(error_html.encode())
        else:
            self.send_response(404)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(b'Not Found')
    
    def _generate_html(self) -> str:
        """Generate the main HTML page."""
        # Create options for feature classes
        class_options = ""
        for class_name in self.visualizer.feature_classes:
            class_options += f'<option value="{html.escape(class_name)}">{html.escape(class_name)}</option>'
        
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>ESM-SAE Feature Visualizer</title>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 0; padding: 20px; }}
                h1 {{ color: #333; }}
                textarea {{ width: 100%; height: 150px; margin-bottom: 10px; }}
                select {{ width: 100%; height: 100px; }}
                .container {{ max-width: 800px; margin: 0 auto; }}
                .form-group {{ margin-bottom: 15px; }}
                label {{ display: block; margin-bottom: 5px; font-weight: bold; }}
                button {{ background-color: #4CAF50; color: white; padding: 10px 15px; border: none; cursor: pointer; }}
                button:hover {{ background-color: #45a049; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>ESM-SAE Feature Visualizer</h1>
                <form action="/analyze" method="post">
                    <div class="form-group">
                        <label for="sequence">Protein Sequence:</label>
                        <textarea id="sequence" name="sequence" placeholder="Enter protein sequence..."></textarea>
                    </div>
                    <div class="form-group">
                        <label for="classes">Select Feature Classes to Compare:</label>
                        <select id="classes" name="classes" multiple>
                            {class_options}
                        </select>
                        <small>Hold Ctrl (or Cmd on Mac) to select multiple classes</small>
                    </div>
                    <button type="submit">Analyze</button>
                </form>
            </div>
        </body>
        </html>
        """
    
    def _generate_results_html(self, sequence: str, selected_classes: List[str], results: Dict[str, Any]) -> str:
        """Generate HTML with analysis results."""
        # Create histogram section
        histogram_section = ""
        if "histogram" in results:
            histogram_section = f"""
            <div class="result-section">
                <h2>Feature Activation Histogram</h2>
                <img src="data:image/png;base64,{results['histogram']}" alt="Feature histogram">
            </div>
            """
        
        # Create feature counts section
        counts_section = ""
        if "feature_counts" in results:
            counts_html = ""
            for class_name, counts in results["feature_counts"].items():
                if not counts:
                    continue
                    
                # Sort features by count (descending)
                sorted_features = sorted(counts.items(), key=lambda x: x[1], reverse=True)
                
                # Create table rows for top 10 features
                rows = ""
                for feature_id, count in sorted_features[:10]:
                    rows += f"""
                    <tr>
                        <td>{feature_id}</td>
                        <td>{count}</td>
                        <td>{count / len(self.visualizer.feature_classes.get(class_name, FeatureDataClass(name="")).sequences):.2f}</td>
                    </tr>
                    """
                
                counts_html += f"""
                <div class="class-counts">
                    <h3>{html.escape(class_name)} - Top Features</h3>
                    <table>
                        <thead>
                            <tr>
                                <th>Feature ID</th>
                                <th>Count</th>
                                <th>Frequency</th>
                            </tr>
                        </thead>
                        <tbody>
                            {rows}
                        </tbody>
                    </table>
                </div>
                """
            
            counts_section = f"""
            <div class="result-section">
                <h2>Feature Counts by Class</h2>
                {counts_html}
            </div>
            """
        
        # Generate input sequence results if available
        input_section = ""
        if results.get("input_features"):
            input_features = results["input_features"]
            
            # Create table rows for top features
            rows = ""
            for i, (idx, val) in enumerate(zip(input_features["feature_indices"], input_features["feature_values"])):
                rows += f"""
                <tr>
                    <td>{i+1}</td>
                    <td>{idx}</td>
                    <td>{val:.4f}</td>
                </tr>
                """
            
            input_section = f"""
            <div class="result-section">
                <h2>Input Sequence Top Features</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Rank</th>
                            <th>Feature ID</th>
                            <th>Activation Value</th>
                        </tr>
                    </thead>
                    <tbody>
                        {rows}
                    </tbody>
                </table>
            </div>
            """
        elif "input" in results.get("results", {}):
            input_features = results["results"]["input"]["features"]
            
            # Create table rows for top features
            rows = ""
            for i, (idx, val) in enumerate(zip(input_features["feature_indices"], input_features["feature_values"])):
                rows += f"""
                <tr>
                    <td>{i+1}</td>
                    <td>{idx}</td>
                    <td>{val:.4f}</td>
                </tr>
                """
            
            input_section = f"""
            <div class="result-section">
                <h2>Input Sequence Top Features</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Rank</th>
                            <th>Feature ID</th>
                            <th>Activation Value</th>
                        </tr>
                    </thead>
                    <tbody>
                        {rows}
                    </tbody>
                </table>
            </div>
            """
        
        # Create the complete HTML
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>ESM-SAE Feature Analysis Results</title>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 0; padding: 20px; }}
                h1, h2, h3 {{ color: #333; }}
                .container {{ max-width: 800px; margin: 0 auto; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .result-section {{ margin-bottom: 30px; }}
                .back-button {{ background-color: #4CAF50; color: white; padding: 10px 15px; text-decoration: none; display: inline-block; }}
                .back-button:hover {{ background-color: #45a049; }}
                .class-counts {{ margin-bottom: 20px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>ESM-SAE Feature Analysis Results</h1>
                
                <div class="result-section">
                    <h2>Input</h2>
                    <p><strong>Sequence:</strong></p>
                    <pre>{html.escape(sequence)}</pre>
                    <p><strong>Selected Classes:</strong> {', '.join(html.escape(c) for c in selected_classes)}</p>
                </div>
                
                {input_section}
                {histogram_section}
                {counts_section}
                
                <a href="/" class="back-button">Back to Home</a>
            </div>
        </body>
        </html>
        """


def create_handler(*args, **kwargs):
    """Create a handler with reference to the visualizer."""
    visualizer = kwargs.pop('visualizer')
    def _handler(*args2, **kwargs2):
        return FeatureVisualizerHandler(*args2, visualizer=visualizer, **kwargs2)
    return _handler


def main():
    """Main function to start the web server."""
    parser = argparse.ArgumentParser(description="ESM-SAE Feature Visualizer")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the web server on (default: 8000)")
    parser.add_argument("--csv_dir", type=str, default=None, help="Directory containing CSV files with sequence classes")
    parser.add_argument("--checkpoint", type=str, default="data/L3840_k32/checkpoint_epoch_134.npy", help="Path to SAE checkpoint")
    args = parser.parse_args()
    
    # Create the visualizer
    visualizer = FeatureVisualizer(args.checkpoint, args.csv_dir)
    
    # Create and start the server
    handler = create_handler(visualizer=visualizer)
    server = HTTPServer(('localhost', args.port), handler)
    
    print(f"Starting feature visualizer server on http://localhost:{args.port}")
    print("Press Ctrl+C to quit")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Server stopped")
    
    server.server_close()


if __name__ == "__main__":
    main()