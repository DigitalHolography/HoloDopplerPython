from __future__ import annotations

import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

import sv_ttk

from holodoppler.Holodoppler import cupy_backend_status
from holodoppler.config import (
    available_builtin_settings,
    export_builtin_setting,
    load_builtin_parameters,
    parameter_definitions,
    ProcessingParameters,
)
from holodoppler.runner import BatchProcessingSummary, process_inputs


class ParameterForm(ttk.Frame):
    def __init__(self, master: tk.Misc) -> None:
        super().__init__(master)
        self._variables: dict[str, tk.Variable] = {}
        self._definitions = parameter_definitions()

        self._canvas = tk.Canvas(
            self,
            highlightthickness=0,
            borderwidth=0,
        )
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=self._canvas.yview)
        self._content = ttk.Frame(self._canvas, padding=12)

        self._content.bind(
            "<Configure>",
            lambda event: self._canvas.configure(scrollregion=self._canvas.bbox("all")),
        )
        self._canvas.bind(
            "<Configure>",
            lambda event: self._canvas.itemconfigure(content_window, width=event.width),
        )

        content_window = self._canvas.create_window((0, 0), window=self._content, anchor="nw")
        self._canvas.configure(yscrollcommand=scrollbar.set)

        self._canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        for row_index, definition in enumerate(self._definitions):
            ttk.Label(self._content, text=definition.label).grid(
                row=row_index,
                column=0,
                sticky="w",
                padx=(0, 12),
                pady=4,
            )

            variable: tk.Variable
            widget: ttk.Widget

            if definition.value_type is bool:
                variable = tk.BooleanVar(value=False)
                widget = ttk.Checkbutton(self._content, variable=variable)
            elif definition.choices:
                variable = tk.StringVar()
                widget = ttk.Combobox(
                    self._content,
                    textvariable=variable,
                    values=definition.choices,
                    state="readonly",
                    width=24,
                )
            else:
                variable = tk.StringVar()
                widget = ttk.Entry(self._content, textvariable=variable, width=26)

            widget.grid(row=row_index, column=1, sticky="ew", pady=4)
            self._variables[definition.name] = variable

        self._content.columnconfigure(1, weight=1)
        self._install_mousewheel_support()
        self.set_parameters(load_builtin_parameters())

    def _install_mousewheel_support(self) -> None:
        self.bind_all("<MouseWheel>", self._on_mousewheel, add="+")
        self.bind_all("<Button-4>", self._on_mousewheel, add="+")
        self.bind_all("<Button-5>", self._on_mousewheel, add="+")

    def _is_descendant(self, widget: tk.Misc | None) -> bool:
        current = widget
        while current is not None:
            if current is self:
                return True
            current = current.master
        return False

    def _on_mousewheel(self, event: tk.Event) -> None:
        hovered_widget = self.winfo_containing(self.winfo_pointerx(), self.winfo_pointery())
        if not self._is_descendant(hovered_widget):
            return

        if getattr(event, "num", None) == 4:
            step = -1
        elif getattr(event, "num", None) == 5:
            step = 1
        else:
            delta = getattr(event, "delta", 0)
            if delta == 0:
                return
            if abs(delta) >= 120:
                step = -int(delta / 120)
            else:
                step = -1 if delta > 0 else 1

        self._canvas.yview_scroll(step, "units")

    def set_parameters(self, parameters: ProcessingParameters) -> None:
        parameter_dict = parameters.to_dict()
        for definition in self._definitions:
            value = parameter_dict[definition.name]
            self._variables[definition.name].set(value)

    def get_parameters(self) -> ProcessingParameters:
        raw_values = {}
        for definition in self._definitions:
            raw_values[definition.name] = self._variables[definition.name].get()
        return ProcessingParameters.from_mapping(raw_values)


class HolodopplerApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()

        self.title("Holodoppler")
        self.geometry("1080x760")
        self.minsize(960, 680)

        sv_ttk.set_theme("dark")

        self._worker_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self._worker_thread: threading.Thread | None = None

        self.input_mode_var = tk.StringVar(value="file")
        self.input_path_var = tk.StringVar()
        self.output_path_var = tk.StringVar()
        self.parameter_source_var = tk.StringVar(value="builtin")
        self.json_path_var = tk.StringVar()
        self.backend_var = tk.StringVar(value="numpy")
        self.pipeline_version_var = tk.StringVar(value="latest")
        self.status_var = tk.StringVar(value="Ready")
        self._cupy_backend_ready, self._cupy_backend_message = cupy_backend_status()

        builtin_settings = available_builtin_settings()
        self.builtin_setting_var = tk.StringVar(
            value=builtin_settings[0] if builtin_settings else "default_parameters.json"
        )

        self._build_layout()
        if not self._cupy_backend_ready and self._cupy_backend_message:
            self._append_log(f"CuPy unavailable: {self._cupy_backend_message}")
        self._update_parameter_source_state()
        self.after(150, self._poll_worker_queue)

    def _build_layout(self) -> None:
        root = ttk.Frame(self, padding=16)
        root.pack(fill="both", expand=True)

        title_frame = ttk.Frame(root)
        title_frame.pack(fill="x")
        ttk.Label(
            title_frame,
            text="Holodoppler Runner",
            font=("Segoe UI", 18, "bold"),
        ).pack(anchor="w")
        ttk.Label(
            title_frame,
            text="Processes .holo files directly, recursively through folders, or from zip archives.",
            font=("Segoe UI", 11),
        ).pack(anchor="w", pady=(4, 0))

        notebook = ttk.Notebook(root)
        notebook.pack(fill="both", expand=True, pady=(16, 0))

        run_tab = ttk.Frame(notebook, padding=16)
        advanced_tab = ttk.Frame(notebook, padding=16)
        notebook.add(run_tab, text="Run")
        notebook.add(advanced_tab, text="Advanced")

        self._build_run_tab(run_tab)
        self._build_advanced_tab(advanced_tab)

        status_bar = ttk.Label(
            root,
            textvariable=self.status_var,
            anchor="w",
        )
        status_bar.pack(fill="x", pady=(12, 0))

    def _build_run_tab(self, parent: ttk.Frame) -> None:
        input_frame = ttk.LabelFrame(parent, text="Input and Output", padding=12)
        input_frame.pack(fill="x")

        ttk.Label(input_frame, text="Input type").grid(row=0, column=0, sticky="w", pady=4)
        input_mode = ttk.Combobox(
            input_frame,
            textvariable=self.input_mode_var,
            values=("file", "folder", "zip"),
            state="readonly",
            width=18,
        )
        input_mode.grid(row=0, column=1, sticky="w", pady=4)

        ttk.Label(input_frame, text="Input path").grid(row=1, column=0, sticky="w", pady=4)
        ttk.Entry(input_frame, textvariable=self.input_path_var).grid(row=1, column=1, sticky="ew", pady=4)
        ttk.Button(input_frame, text="Browse...", command=self._browse_input).grid(row=1, column=2, padx=(8, 0))

        ttk.Label(input_frame, text="Output folder").grid(row=2, column=0, sticky="w", pady=4)
        ttk.Entry(input_frame, textvariable=self.output_path_var).grid(row=2, column=1, sticky="ew", pady=4)
        ttk.Button(input_frame, text="Browse...", command=self._browse_output).grid(row=2, column=2, padx=(8, 0))

        input_frame.columnconfigure(1, weight=1)

        parameters_frame = ttk.LabelFrame(parent, text="Parameters", padding=12)
        parameters_frame.pack(fill="x", pady=(16, 0))

        ttk.Radiobutton(
            parameters_frame,
            text="Builtin setting",
            value="builtin",
            variable=self.parameter_source_var,
            command=self._update_parameter_source_state,
        ).grid(row=0, column=0, sticky="w", pady=4)
        ttk.Radiobutton(
            parameters_frame,
            text="JSON file",
            value="json",
            variable=self.parameter_source_var,
            command=self._update_parameter_source_state,
        ).grid(row=1, column=0, sticky="w", pady=4)
        ttk.Radiobutton(
            parameters_frame,
            text="Advanced tab values",
            value="advanced",
            variable=self.parameter_source_var,
            command=self._update_parameter_source_state,
        ).grid(row=2, column=0, sticky="w", pady=4)

        self.builtin_setting_combo = ttk.Combobox(
            parameters_frame,
            textvariable=self.builtin_setting_var,
            values=available_builtin_settings(),
            state="readonly",
            width=28,
        )
        self.builtin_setting_combo.grid(row=0, column=1, sticky="w", padx=(12, 0), pady=4)

        json_frame = ttk.Frame(parameters_frame)
        json_frame.grid(row=1, column=1, sticky="ew", padx=(12, 0), pady=4)
        self.json_entry = ttk.Entry(json_frame, textvariable=self.json_path_var)
        self.json_entry.pack(side="left", fill="x", expand=True)
        self.json_browse_button = ttk.Button(json_frame, text="Browse...", command=self._browse_json)
        self.json_browse_button.pack(side="left", padx=(8, 0))
        parameters_frame.columnconfigure(1, weight=1)

        runtime_frame = ttk.LabelFrame(parent, text="Runtime", padding=12)
        runtime_frame.pack(fill="x", pady=(16, 0))

        ttk.Label(runtime_frame, text="Backend").grid(row=0, column=0, sticky="w", padx=(0, 10), pady=4)
        backend_combo = ttk.Combobox(
            runtime_frame,
            textvariable=self.backend_var,
            values=("numpy", "cupy") if self._cupy_backend_ready else ("numpy",),
            state="readonly",
            width=18,
        )
        backend_combo.grid(row=0, column=1, sticky="w", padx=(0, 28), pady=4)

        ttk.Label(runtime_frame, text="Pipeline version").grid(row=0, column=2, sticky="w", padx=(0, 10), pady=4)
        pipeline_combo = ttk.Combobox(
            runtime_frame,
            textvariable=self.pipeline_version_var,
            values=("latest", "old"),
            state="readonly",
            width=18,
        )
        pipeline_combo.grid(row=0, column=3, sticky="w", padx=(0, 16), pady=4)

        self.run_button = ttk.Button(runtime_frame, text="Run Processing", command=self._run_processing)
        self.run_button.grid(row=0, column=4, sticky="e", padx=(24, 0))
        runtime_frame.columnconfigure(4, weight=1)

        if not self._cupy_backend_ready and self._cupy_backend_message:
            ttk.Label(
                runtime_frame,
                text="CuPy is unavailable in this environment. NumPy processing remains available.",
                wraplength=640,
            ).grid(row=1, column=0, columnspan=5, sticky="w", pady=(8, 0))

        log_frame = ttk.LabelFrame(parent, text="Log", padding=12)
        log_frame.pack(fill="both", expand=True, pady=(16, 0))
        self.log_widget = ScrolledText(log_frame, height=18, wrap="word")
        self.log_widget.pack(fill="both", expand=True)
        self.log_widget.configure(state="disabled")

    def _build_advanced_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(1, weight=1)
        parent.rowconfigure(0, weight=1)

        library_frame = ttk.LabelFrame(parent, text="Settings Library", padding=12)
        library_frame.grid(row=0, column=0, sticky="nsw", padx=(0, 16))

        self.library_tree = ttk.Treeview(library_frame, columns=("name",), show="headings", height=10)
        self.library_tree.heading("name", text="Preset")
        self.library_tree.column("name", width=220, stretch=False)
        self.library_tree.pack(fill="both", expand=True)

        for setting_name in available_builtin_settings():
            self.library_tree.insert("", "end", values=(setting_name,))

        buttons_frame = ttk.Frame(library_frame)
        buttons_frame.pack(fill="x", pady=(12, 0))
        ttk.Button(buttons_frame, text="Load Preset", command=self._load_selected_preset).pack(fill="x")
        ttk.Button(buttons_frame, text="Export Preset...", command=self._export_selected_preset).pack(fill="x", pady=(8, 0))
        ttk.Button(buttons_frame, text="Load JSON...", command=self._load_json_into_form).pack(fill="x", pady=(8, 0))
        ttk.Button(buttons_frame, text="Save Current As...", command=self._save_form_as_json).pack(fill="x", pady=(8, 0))
        ttk.Button(buttons_frame, text="Reset To Default", command=self._reset_form_to_default).pack(fill="x", pady=(8, 0))

        form_frame = ttk.LabelFrame(parent, text="Advanced Parameters", padding=0)
        form_frame.grid(row=0, column=1, sticky="nsew")
        form_frame.columnconfigure(0, weight=1)
        form_frame.rowconfigure(0, weight=1)

        self.parameter_form = ParameterForm(form_frame)
        self.parameter_form.grid(row=0, column=0, sticky="nsew")

    def _append_log(self, message: str) -> None:
        self.log_widget.configure(state="normal")
        self.log_widget.insert("end", message + "\n")
        self.log_widget.see("end")
        self.log_widget.configure(state="disabled")

    def _selected_library_setting(self) -> str | None:
        selection = self.library_tree.selection()
        if not selection:
            return None
        values = self.library_tree.item(selection[0], "values")
        return values[0] if values else None

    def _load_selected_preset(self) -> None:
        selected = self._selected_library_setting()
        if not selected:
            messagebox.showinfo("Settings Library", "Select a preset first.")
            return
        self.parameter_form.set_parameters(load_builtin_parameters(selected))

    def _export_selected_preset(self) -> None:
        selected = self._selected_library_setting()
        if not selected:
            messagebox.showinfo("Settings Library", "Select a preset first.")
            return
        destination = filedialog.asksaveasfilename(
            title="Export Preset",
            defaultextension=".json",
            initialfile=selected,
            filetypes=[("JSON files", "*.json")],
        )
        if destination:
            export_builtin_setting(selected, destination)

    def _load_json_into_form(self) -> None:
        file_path = filedialog.askopenfilename(
            title="Load Parameters JSON",
            filetypes=[("JSON files", "*.json")],
        )
        if not file_path:
            return
        try:
            self.parameter_form.set_parameters(ProcessingParameters.from_json_file(file_path))
        except Exception as exc:
            messagebox.showerror("Invalid Parameters", str(exc))

    def _save_form_as_json(self) -> None:
        try:
            parameters = self.parameter_form.get_parameters()
        except Exception as exc:
            messagebox.showerror("Invalid Parameters", str(exc))
            return

        file_path = filedialog.asksaveasfilename(
            title="Save Parameters JSON",
            defaultextension=".json",
            initialfile="parameters.json",
            filetypes=[("JSON files", "*.json")],
        )
        if file_path:
            parameters.save_json(file_path)

    def _reset_form_to_default(self) -> None:
        self.parameter_form.set_parameters(load_builtin_parameters())

    def _browse_input(self) -> None:
        input_mode = self.input_mode_var.get()
        selected_path = ""
        if input_mode == "file":
            selected_path = filedialog.askopenfilename(
                title="Select .holo File",
                filetypes=[("Holo files", "*.holo")],
            )
        elif input_mode == "folder":
            selected_path = filedialog.askdirectory(title="Select Folder")
        elif input_mode == "zip":
            selected_path = filedialog.askopenfilename(
                title="Select Zip Archive",
                filetypes=[("Zip archives", "*.zip")],
            )
        if selected_path:
            self.input_path_var.set(selected_path)

    def _browse_output(self) -> None:
        selected_path = filedialog.askdirectory(title="Select Output Folder")
        if selected_path:
            self.output_path_var.set(selected_path)

    def _browse_json(self) -> None:
        selected_path = filedialog.askopenfilename(
            title="Select Parameters JSON",
            filetypes=[("JSON files", "*.json")],
        )
        if selected_path:
            self.json_path_var.set(selected_path)

    def _update_parameter_source_state(self) -> None:
        source_mode = self.parameter_source_var.get()
        builtin_state = "readonly" if source_mode == "builtin" else "disabled"
        json_state = "normal" if source_mode == "json" else "disabled"

        self.builtin_setting_combo.configure(state=builtin_state)
        self.json_entry.configure(state=json_state)
        self.json_browse_button.configure(state=json_state)

    def _resolve_parameters(self) -> ProcessingParameters:
        source_mode = self.parameter_source_var.get()
        if source_mode == "builtin":
            return load_builtin_parameters(self.builtin_setting_var.get())
        if source_mode == "json":
            json_path = self.json_path_var.get().strip()
            if not json_path:
                raise ValueError("Select a parameters JSON file.")
            return ProcessingParameters.from_json_file(json_path)
        return self.parameter_form.get_parameters()

    def _run_processing(self) -> None:
        if self._worker_thread and self._worker_thread.is_alive():
            return

        input_path = self.input_path_var.get().strip()
        output_path = self.output_path_var.get().strip()
        if not input_path:
            messagebox.showerror("Missing Input", "Select an input path first.")
            return
        if not output_path:
            messagebox.showerror("Missing Output", "Select an output folder first.")
            return

        try:
            parameters = self._resolve_parameters()
        except Exception as exc:
            messagebox.showerror("Invalid Parameters", str(exc))
            return

        self.run_button.configure(state="disabled")
        self.status_var.set("Processing")
        self._append_log(f"Starting processing for {input_path}")
        backend = self.backend_var.get()
        pipeline_version = self.pipeline_version_var.get()

        self._worker_thread = threading.Thread(
            target=self._worker_process,
            args=(input_path, output_path, parameters, backend, pipeline_version),
            daemon=True,
        )
        self._worker_thread.start()

    def _worker_process(
        self,
        input_path: str,
        output_path: str,
        parameters: ProcessingParameters,
        backend: str,
        pipeline_version: str,
    ) -> None:
        try:
            summary = process_inputs(
                input_path=input_path,
                output_root=output_path,
                parameters=parameters,
                backend=backend,
                pipeline_version=pipeline_version,
                progress=lambda message: self._worker_queue.put(("log", message)),
            )
            self._worker_queue.put(("done", summary))
        except Exception as exc:
            self._worker_queue.put(("error", str(exc)))

    def _poll_worker_queue(self) -> None:
        while True:
            try:
                event_type, payload = self._worker_queue.get_nowait()
            except queue.Empty:
                break

            if event_type == "log":
                self._append_log(str(payload))
                self.status_var.set(str(payload))
            elif event_type == "error":
                self.run_button.configure(state="normal")
                self.status_var.set("Failed")
                self._append_log(f"Error: {payload}")
                messagebox.showerror("Processing Failed", str(payload))
            elif event_type == "done":
                summary = payload
                self.run_button.configure(state="normal")
                self._handle_completed_summary(summary)

        self.after(150, self._poll_worker_queue)

    def _handle_completed_summary(self, summary: object) -> None:
        if not isinstance(summary, BatchProcessingSummary):
            self.status_var.set("Completed")
            return

        processed_count = len(summary.processed)
        failed_count = len(summary.failed)
        self.status_var.set(f"Completed: {processed_count} processed, {failed_count} failed")
        self._append_log(f"Completed: {processed_count} processed, {failed_count} failed")

        if summary.failed:
            failures = "\n".join(f"{item.source_label}: {item.error}" for item in summary.failed)
            messagebox.showwarning("Processing Completed With Failures", failures)
        else:
            messagebox.showinfo("Processing Completed", f"Processed {processed_count} item(s).")


def launch_app() -> None:
    app = HolodopplerApp()
    app.mainloop()


if __name__ == "__main__":
    launch_app()
