import os
import json
import shutil
from datetime import datetime
from logger_utils import logger

class ProjectManager:
    def __init__(self, projects_dir="projects"):
        self.projects_dir = projects_dir
        os.makedirs(projects_dir, exist_ok=True)

    def create_project(self, name, video_path, identity_map, settings):
        """Saves a transformation setup as a reusable project file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        project_id = f"{name}_{timestamp}"
        project_path = os.path.join(self.projects_dir, f"{project_id}.avt")
        
        project_data = {
            "name": name,
            "id": project_id,
            "video_path": os.path.abspath(video_path),
            "identity_map": identity_map,
            "settings": settings,
            "created_at": str(datetime.now())
        }
        
        with open(project_path, 'w') as f:
            json.dump(project_data, f, indent=4)
            
        logger.info(f"Project '{name}' saved to {project_path}")
        return project_path

    def list_projects(self):
        projects = []
        for f in os.listdir(self.projects_dir):
            if f.endswith(".avt"):
                with open(os.path.join(self.projects_dir, f), 'r') as file:
                    projects.append(json.load(file))
        return projects

    def load_project(self, project_id):
        path = os.path.join(self.projects_dir, f"{project_id}.avt")
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
        return None

if __name__ == "__main__":
    pass
