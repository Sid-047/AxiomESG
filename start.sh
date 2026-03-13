#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
#  AxiomESG — start.sh
#  One-shot launcher + helper commands for the Docker stack.
# ─────────────────────────────────────────────────────────────
set -e

PROJECT="axiomesg"
COMPOSE="docker compose"

# ── colours ──────────────────────────────────────────────────
BOLD="\033[1m"
DIM="\033[2m"
CYAN="\033[36m"
GREEN="\033[32m"
YELLOW="\033[33m"
RED="\033[31m"
RESET="\033[0m"

banner() {
  echo ""
  echo -e "${BOLD}${CYAN}  ╔═══════════════════════════════════════════╗${RESET}"
  echo -e "${BOLD}${CYAN}  ║            A X I O M  E S G              ║${RESET}"
  echo -e "${BOLD}${CYAN}  ║     Deterministic ESG Intelligence       ║${RESET}"
  echo -e "${BOLD}${CYAN}  ╚═══════════════════════════════════════════╝${RESET}"
  echo ""
}

usage() {
  echo -e "${BOLD}Usage:${RESET}  ./start.sh ${GREEN}<command>${RESET}"
  echo ""
  echo -e "${BOLD}Commands:${RESET}"
  echo -e "  ${GREEN}up${RESET}            Build & start all services (detached)"
  echo -e "  ${GREEN}up:dev${RESET}        Build & start in foreground (with logs)"
  echo -e "  ${GREEN}down${RESET}          Stop & remove all containers"
  echo -e "  ${GREEN}restart${RESET}       Restart all services"
  echo -e "  ${GREEN}build${RESET}         Rebuild Docker images (no cache)"
  echo -e "  ${GREEN}logs${RESET}          Tail logs from all services"
  echo -e "  ${GREEN}logs:back${RESET}     Tail logs from backend only"
  echo -e "  ${GREEN}logs:front${RESET}    Tail logs from frontend only"
  echo -e "  ${GREEN}status${RESET}        Show container status"
  echo -e "  ${GREEN}shell:back${RESET}    Open a shell in the backend container"
  echo -e "  ${GREEN}shell:front${RESET}   Open a shell in the frontend container"
  echo -e "  ${GREEN}test${RESET}          Run backend pytest suite"
  echo -e "  ${GREEN}health${RESET}        Check backend health endpoint"
  echo -e "  ${GREEN}algorithms${RESET}    List available ESG algorithms"
  echo -e "  ${GREEN}clean${RESET}         Remove containers, images & volumes"
  echo ""
}

ensure_env() {
  if [ ! -f "./backend/.env" ]; then
    echo -e "${YELLOW}⚠  No backend/.env found — creating a placeholder.${RESET}"
    cat > ./backend/.env <<'EOF'
# AxiomESG Backend Environment
# Choose your LLM provider and fill in the keys.

LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=
OPENROUTER_MODEL=openrouter/auto

# AZURE_OPENAI_ENDPOINT=
# AZURE_OPENAI_API_KEY=
# AZURE_OPENAI_DEPLOYMENT=
# AZURE_OPENAI_API_VERSION=2024-02-15-preview

# GEMINI_API_KEY=
# GEMINI_MODEL=gemini-1.5-flash

# Optional OCR
# AZURE_DOCINTEL_ENDPOINT=
# AZURE_DOCINTEL_KEY=

# Algorithm default
DEFAULT_ALGORITHM=heuristic

# CORS
CORS_ORIGINS=http://localhost:3000

# Redis (optional)
# REDIS_URL=
EOF
    echo -e "${DIM}   Created backend/.env — edit it before running.${RESET}"
  fi
}

# ── commands ─────────────────────────────────────────────────

cmd_up() {
  ensure_env
  echo -e "${CYAN}▸ Building & starting services...${RESET}"
  $COMPOSE up -d --build
  echo ""
  echo -e "${GREEN}✓ AxiomESG is running${RESET}"
  echo -e "  Backend  → ${BOLD}http://localhost:8000${RESET}"
  echo -e "  Frontend → ${BOLD}http://localhost:3000${RESET}"
  echo ""
  echo -e "${DIM}  Run ${RESET}${GREEN}./start.sh logs${RESET}${DIM} to tail logs.${RESET}"
}

cmd_up_dev() {
  ensure_env
  echo -e "${CYAN}▸ Starting in dev mode (foreground)...${RESET}"
  $COMPOSE up --build
}

cmd_down() {
  echo -e "${CYAN}▸ Stopping services...${RESET}"
  $COMPOSE down
  echo -e "${GREEN}✓ All services stopped.${RESET}"
}

cmd_restart() {
  echo -e "${CYAN}▸ Restarting services...${RESET}"
  $COMPOSE restart
  echo -e "${GREEN}✓ Restarted.${RESET}"
}

cmd_build() {
  echo -e "${CYAN}▸ Building images (no cache)...${RESET}"
  $COMPOSE build --no-cache
  echo -e "${GREEN}✓ Build complete.${RESET}"
}

cmd_logs() {
  $COMPOSE logs -f --tail=100
}

cmd_logs_back() {
  $COMPOSE logs -f --tail=100 backend
}

cmd_logs_front() {
  $COMPOSE logs -f --tail=100 frontend
}

cmd_status() {
  echo -e "${CYAN}▸ Container status${RESET}"
  $COMPOSE ps
}

cmd_shell_back() {
  echo -e "${CYAN}▸ Opening backend shell...${RESET}"
  docker exec -it axiomesg-backend /bin/bash
}

cmd_shell_front() {
  echo -e "${CYAN}▸ Opening frontend shell...${RESET}"
  docker exec -it axiomesg-frontend /bin/sh
}

cmd_test() {
  echo -e "${CYAN}▸ Running backend tests...${RESET}"
  docker exec -it axiomesg-backend python -m pytest tests/ -v
}

cmd_health() {
  echo -e "${CYAN}▸ Checking backend health...${RESET}"
  curl -s http://localhost:8000/ | python3 -m json.tool 2>/dev/null || echo -e "${RED}✗ Backend not reachable.${RESET}"
}

cmd_algorithms() {
  echo -e "${CYAN}▸ Available algorithms:${RESET}"
  curl -s http://localhost:8000/api/algorithms | python3 -m json.tool 2>/dev/null || echo -e "${RED}✗ Backend not reachable.${RESET}"
}

cmd_clean() {
  echo -e "${RED}▸ Removing containers, images & volumes...${RESET}"
  $COMPOSE down --rmi all --volumes --remove-orphans
  echo -e "${GREEN}✓ Cleaned.${RESET}"
}

# ── main ─────────────────────────────────────────────────────

banner

case "${1:-up}" in
  up)           cmd_up ;;
  up:dev)       cmd_up_dev ;;
  down)         cmd_down ;;
  restart)      cmd_restart ;;
  build)        cmd_build ;;
  logs)         cmd_logs ;;
  logs:back)    cmd_logs_back ;;
  logs:front)   cmd_logs_front ;;
  status)       cmd_status ;;
  shell:back)   cmd_shell_back ;;
  shell:front)  cmd_shell_front ;;
  test)         cmd_test ;;
  health)       cmd_health ;;
  algorithms)   cmd_algorithms ;;
  clean)        cmd_clean ;;
  help)         usage ;;
  *)            usage ;;
esac
