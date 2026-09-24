import traceback
 
 from rich.console import Console
+from rich.markup import escape
 from rich.panel import Panel
...
         if pattern in exc_str or pattern in exc_type:
             error_msg = short_msg or exc_str
-            console.print(f"\n[bold red]Error:[/] {error_msg}")
+            console.print(f"\n[bold red]Error:[/] {escape(error_msg)}")
             console.print(f"[green]Fix:[/] {fix}")
...
     # Unknown error — show type + message
-    console.print(f"\n[bold red]Error:[/] {exc_type}: {exc_str}")
+    console.print(f"\n[bold red]Error:[/] {escape(exc_type)}: {escape(exc_str)}")
     console.print("[dim]Run with --verbose for the full traceback.[/]")