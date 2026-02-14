    def _on_project_selected(self, event):
        """Handle project selection - load config AND PRD"""
        selection = self.projects_tree.selection()
        if not selection:
            return

        item = selection[0]
        project_name = self.projects_tree.item(item, 'text')
        values = self.projects_tree.item(item, 'values')

        self.current_project = {
            'name': project_name,
            'type': values[0] if len(values) > 0 else 'API',
            'domain': values[1] if len(values) > 1 else '',
            'framework': values[2] if len(values) > 2 else '',
        }
        self.current_project_id = project_name

        # Load config AND PRD
        projects_list = self.orchestrator.list_projects()
        for proj in projects_list:
            if proj['name'] == project_name:
                project_path = Path(proj['path'])

                # Load config.json
                config_file = project_path / 'config.json'
                if config_file.exists():
                    with open(config_file, 'r') as f:
                        config_data = json.load(f)
                        self.orchestrator.current_config = ProjectConfig.from_dict(config_data)
                        self.orchestrator.current_project_dir = project_path

                # Load prd.json if exists - CRITICAL FIX for execution
                prd_file = project_path / 'prd.json'
                if prd_file.exists():
                    with open(prd_file, 'r') as f:
                        self.current_prd = json.load(f)
                        logger.info(f"📋 Loaded PRD with {self.current_prd.get('total_items', 0)} items")
                else:
                    # Fallback: try to build PRD from metadata prd_backlog
                    metadata = config_data.get('metadata', {})
                    if metadata.get('prd_backlog'):
                        self.current_prd = {
                            'backlog': metadata['prd_backlog'].get('backlog', []),
                            'total_items': len(metadata['prd_backlog'].get('backlog', [])),
                            'project_id': project_name,
                        }
                        logger.info(f"📋 Built PRD from metadata with {self.current_prd['total_items']} items")
                    else:
                        self.current_prd = None
                        logger.info("⚠️ No PRD found for this project")

                break

        logger.info(f"📁 Selected project: {project_name}")

        # Sync refinement text box with project context
        if hasattr(self, 'task_text'):
            try:
                placeholder = build_project_refinement_placeholder_text(self.current_project)
                self.task_text.delete('1.0', 'end')
                self.task_text.insert('1.0', placeholder)
            except Exception:
                pass

        # Jump directly to refinement tab for better UX
        try:
            self.notebook.select(get_default_refinement_tab_index())
        except Exception:
            pass

        prd_status = f" ({self.current_prd.get('total_items', 0)} PRD items ready)" if self.current_prd else " (no PRD - generate first)"
        messagebox.showinfo(
            "Project Loaded",
            f"Project '{project_name}' loaded{prd_status}.\n\nGo to Execution tab to run agents, or use Edit Project to modify."
        )
