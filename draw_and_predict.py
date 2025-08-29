import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import time
import threading
from inference_stroke import StrokeInference


class StrokeDrawingApp:
    """Interactive drawing app for stroke-based mathematical expression recognition"""
    
    def __init__(self, config_path, checkpoint_path):
        # Initialize inference engine in background
        self.inference = None
        self.loading = True
        
        # Create GUI first
        self.setup_gui()
        
        # Load model in background thread
        self.load_model_thread = threading.Thread(
            target=self.load_inference_engine, 
            args=(config_path, checkpoint_path)
        )
        self.load_model_thread.daemon = True
        self.load_model_thread.start()
        
        # Drawing state
        self.current_stroke = []
        self.all_strokes = []
        self.drawing = False
        self.stroke_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
        
    def setup_gui(self):
        """Create the GUI interface"""
        self.root = tk.Tk()
        self.root.title("Stroke-Aware Mathematical Expression Recognition")
        self.root.geometry("800x600")
        
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title = ttk.Label(main_frame, text="Draw Mathematical Expression", 
                         font=('Arial', 16, 'bold'))
        title.grid(row=0, column=0, columnspan=3, pady=(0, 10))
        
        # Status label
        self.status_label = ttk.Label(main_frame, text="🔄 Loading model...", 
                                     font=('Arial', 10))
        self.status_label.grid(row=1, column=0, columnspan=3, pady=(0, 10))
        
        # Drawing canvas
        self.canvas = tk.Canvas(main_frame, width=600, height=300, bg='white', 
                               relief=tk.SUNKEN, borderwidth=2)
        self.canvas.grid(row=2, column=0, columnspan=3, pady=(0, 10))
        
        # Bind canvas events
        self.canvas.bind("<Button-1>", self.start_stroke)
        self.canvas.bind("<B1-Motion>", self.draw_stroke)
        self.canvas.bind("<ButtonRelease-1>", self.end_stroke)
        
        # Control buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=3, column=0, columnspan=3, pady=(10, 0))
        
        self.predict_button = ttk.Button(button_frame, text="🎯 Predict", 
                                        command=self.predict_expression, state='disabled')
        self.predict_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.clear_button = ttk.Button(button_frame, text="🗑️ Clear", 
                                      command=self.clear_canvas)
        self.clear_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.load_inkml_button = ttk.Button(button_frame, text="📁 Load InkML", 
                                           command=self.load_inkml_file, state='disabled')
        self.load_inkml_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Results area
        results_frame = ttk.LabelFrame(main_frame, text="Prediction Results", padding="10")
        results_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # Result text
        self.result_text = tk.Text(results_frame, height=8, width=70, wrap=tk.WORD)
        self.result_text.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        # Scrollbar for results
        scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.result_text.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.result_text.configure(yscrollcommand=scrollbar.set)
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        results_frame.columnconfigure(0, weight=1)
        
        # Instructions
        instructions = """
🖊️ Instructions:
1. Draw mathematical expression with your mouse
2. Each stroke will be shown in a different color
3. Click 'Predict' to get the LaTeX prediction
4. Use 'Clear' to start over
5. Or load an existing InkML file with 'Load InkML'

✨ Tips:
- Draw clearly with distinct strokes
- Try simple expressions like "2", "x+1", "a^2"
- The model was trained on handwritten math expressions
        """.strip()
        
        self.result_text.insert(tk.END, instructions)
        
    def load_inference_engine(self, config_path, checkpoint_path):
        """Load the inference engine in background"""
        try:
            self.root.after(0, lambda: self.status_label.config(text="🔄 Loading model..."))
            
            self.inference = StrokeInference(config_path, checkpoint_path)
            self.loading = False
            
            # Update GUI on main thread
            self.root.after(0, self.on_model_loaded)
            
        except Exception as e:
            self.root.after(0, lambda: self.on_model_load_error(str(e)))
    
    def on_model_loaded(self):
        """Called when model is successfully loaded"""
        self.status_label.config(text="✅ Model loaded! Draw an expression and click Predict.")
        self.predict_button.config(state='normal')
        self.load_inkml_button.config(state='normal')
        
    def on_model_load_error(self, error_msg):
        """Called when model loading fails"""
        self.status_label.config(text=f"❌ Failed to load model: {error_msg}")
        messagebox.showerror("Model Loading Error", f"Failed to load model:\n{error_msg}")
        
    def start_stroke(self, event):
        """Start a new stroke"""
        if self.loading:
            return
            
        self.drawing = True
        self.current_stroke = []
        self.start_time = time.time()
        
        # Add first point
        x, y = event.x, event.y
        t = 0.0
        self.current_stroke.append((x, y, t))
        
    def draw_stroke(self, event):
        """Continue drawing current stroke"""
        if not self.drawing or self.loading:
            return
            
        x, y = event.x, event.y
        t = time.time() - self.start_time
        
        # Add point to current stroke
        self.current_stroke.append((x, y, t))
        
        # Draw line from previous point
        if len(self.current_stroke) > 1:
            prev_x, prev_y, _ = self.current_stroke[-2]
            color = self.stroke_colors[len(self.all_strokes) % len(self.stroke_colors)]
            self.canvas.create_line(prev_x, prev_y, x, y, fill=color, width=2)
            
    def end_stroke(self, event):
        """Finish current stroke"""
        if not self.drawing or self.loading:
            return
            
        self.drawing = False
        
        if len(self.current_stroke) > 1:
            self.all_strokes.append(self.current_stroke.copy())
            
        self.current_stroke = []
        
        # Update status
        self.status_label.config(text=f"✏️ Drew {len(self.all_strokes)} stroke(s). Click Predict when ready.")
        
    def clear_canvas(self):
        """Clear the canvas and reset strokes"""
        self.canvas.delete("all")
        self.all_strokes = []
        self.current_stroke = []
        self.status_label.config(text="✅ Canvas cleared. Draw an expression.")
        
    def predict_expression(self):
        """Make prediction from drawn strokes"""
        if self.loading:
            messagebox.showwarning("Model Loading", "Model is still loading, please wait...")
            return
            
        if not self.all_strokes:
            messagebox.showwarning("No Drawing", "Please draw an expression first!")
            return
            
        try:
            self.status_label.config(text="🔄 Making prediction...")
            self.predict_button.config(state='disabled')
            
            # Convert canvas coordinates to normalized format
            normalized_strokes = self.normalize_canvas_strokes()
            
            # Make prediction
            result = self.inference.predict_from_strokes(normalized_strokes)
            
            # Display results
            self.display_prediction_results(result)
            
        except Exception as e:
            messagebox.showerror("Prediction Error", f"Failed to make prediction:\n{str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            self.predict_button.config(state='normal')
            self.status_label.config(text="✅ Prediction complete!")
            
    def normalize_canvas_strokes(self):
        """Convert canvas pixel coordinates to normalized strokes"""
        if not self.all_strokes:
            return []
            
        # Get canvas dimensions
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        normalized_strokes = []
        for stroke in self.all_strokes:
            normalized_points = []
            for x, y, t in stroke:
                # Normalize to [0, 1] range
                norm_x = x / canvas_width
                norm_y = y / canvas_height
                normalized_points.append((norm_x, norm_y, t))
            normalized_strokes.append(normalized_points)
            
        return normalized_strokes
        
    def display_prediction_results(self, result):
        """Display prediction results in the text area"""
        self.result_text.delete(1.0, tk.END)
        
        results_text = f"""
🎯 PREDICTION RESULTS
{'=' * 50}

📝 Predicted LaTeX: {result['predicted_latex']}

📊 Details:
   • Number of strokes: {len(self.all_strokes)}
   • Valid strokes processed: {result['num_valid_strokes']}
   • Confidence scores (first 10): {result['confidence_scores'][:10]}

💡 Tips:
   • If prediction seems wrong, try drawing more clearly
   • Make sure strokes are distinct and well-formed
   • Simple expressions work better (try "2", "x", "a+b")

{'=' * 50}
        """.strip()
        
        self.result_text.insert(tk.END, results_text)
        
    def load_inkml_file(self):
        """Load and predict from an InkML file"""
        if self.loading:
            messagebox.showwarning("Model Loading", "Model is still loading, please wait...")
            return
            
        file_path = filedialog.askopenfilename(
            title="Select InkML file",
            filetypes=[("InkML files", "*.inkml"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
            
        try:
            self.status_label.config(text="🔄 Processing InkML file...")
            self.load_inkml_button.config(state='disabled')
            
            # Make prediction from InkML
            result = self.inference.predict_from_inkml(file_path)
            
            if result:
                # Display results
                self.result_text.delete(1.0, tk.END)
                
                results_text = f"""
🎯 INKML PREDICTION RESULTS
{'=' * 50}

📁 File: {file_path.split('/')[-1]}

📝 Original Label: {result['original_label']}
🎯 Predicted LaTeX: {result['predicted_latex']}

📊 Details:
   • Valid strokes: {result['num_valid_strokes']}
   • Normalization scale: {result['normalization_info']['scale_factor']:.4f}
   • Confidence scores (first 10): {result['confidence_scores'][:10]}

{'=' * 50}
                """.strip()
                
                self.result_text.insert(tk.END, results_text)
            else:
                messagebox.showerror("InkML Error", "Failed to process InkML file")
                
        except Exception as e:
            messagebox.showerror("InkML Error", f"Failed to process InkML:\n{str(e)}")
        finally:
            self.load_inkml_button.config(state='normal')
            self.status_label.config(text="✅ InkML processing complete!")
            
    def run(self):
        """Start the application"""
        self.root.mainloop()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Interactive Stroke Drawing and Prediction')
    parser.add_argument('--config', default='config_stroke.yaml',
                       help='Path to stroke config file')
    parser.add_argument('--checkpoint', required=True,
                       help='Path to trained model checkpoint')
    args = parser.parse_args()
    
    if not args.checkpoint:
        print("❌ Please provide checkpoint path with --checkpoint")
        return
        
    # Create and run the app
    app = StrokeDrawingApp(args.config, args.checkpoint)
    app.run()


if __name__ == '__main__':
    main()