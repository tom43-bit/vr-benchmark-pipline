#!/bin/bash
# 文件名: git_push_advanced.sh
# 功能: 高级Git提交脚本，支持选择性提交

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

REPO_PATH="/nfs/xtjin/benchmark"
cd $REPO_PATH || exit 1

show_help() {
    echo "用法: ./git_push.sh [选项]"
    echo "选项:"
    echo "  -m, --message    提交信息"
    echo "  -f, --force      强制推送"
    echo "  -b, --branch     指定分支"
    echo "  -d, --dry-run    试运行（不实际提交）"
    echo "  -h, --help       显示帮助"
}

# 解析参数
FORCE_PUSH=0
DRY_RUN=0
BRANCH=""
COMMIT_MSG=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--message)
            COMMIT_MSG="$2"
            shift 2
            ;;
        -f|--force)
            FORCE_PUSH=1
            shift
            ;;
        -b|--branch)
            BRANCH="$2"
            shift 2
            ;;
        -d|--dry-run)
            DRY_RUN=1
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "未知选项: $1"
            show_help
            exit 1
            ;;
    esac
done

echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Git 高级提交脚本${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "仓库: $(pwd)"
echo -e "分支: $(git branch --show-current)"
echo -e "时间: $(date)"
echo -e "${BLUE}========================================${NC}\n"

# 1. 检查状态
echo -e "${YELLOW}1. 检查工作区状态...${NC}"
git fetch --all
git status

if [ -z "$(git status --porcelain)" ]; then
    echo -e "\n${GREEN}✓ 没有需要提交的修改${NC}"
    exit 0
fi

# 2. 显示修改详情
echo -e "\n${YELLOW}2. 修改详情:${NC}"
echo -e "${CYAN}新增文件:${NC}"
git ls-files --others --exclude-standard
echo -e "\n${CYAN}修改文件:${NC}"
git diff --name-only
echo -e "\n${CYAN}删除文件:${NC}"
git ls-files --deleted

# 3. 确认提交
echo -e "\n${YELLOW}3. 确认提交? [y/N]${NC}"
read -p "确认: " confirm
if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
    echo -e "${RED}取消提交${NC}"
    exit 0
fi

# 4. 获取提交信息
if [ -z "$COMMIT_MSG" ]; then
    echo -e "\n${YELLOW}4. 请输入提交信息:${NC}"
    read -p "提交信息: " COMMIT_MSG
    if [ -z "$COMMIT_MSG" ]; then
        COMMIT_MSG="自动提交 $(date '+%Y-%m-%d %H:%M:%S')"
    fi
fi

# 5. 添加文件
echo -e "\n${YELLOW}5. 添加文件...${NC}"
git add .

# 6. 提交
echo -e "\n${YELLOW}6. 提交修改...${NC}"
if [ $DRY_RUN -eq 1 ]; then
    echo -e "${YELLOW}[试运行] git commit -m \"$COMMIT_MSG\"${NC}"
else
    git commit -m "$COMMIT_MSG"
    echo -e "${GREEN}✓ 提交完成${NC}"
fi

# 7. 拉取更新
echo -e "\n${YELLOW}7. 拉取远程更新...${NC}"
if [ $DRY_RUN -eq 1 ]; then
    echo -e "${YELLOW}[试运行] git pull --rebase${NC}"
else
    if git pull --rebase; then
        echo -e "${GREEN}✓ 拉取完成${NC}"
    else
        echo -e "${RED}✗ 拉取失败，请手动解决冲突${NC}"
        exit 1
    fi
fi

# 8. 推送
echo -e "\n${YELLOW}8. 推送到远程...${NC}"
PUSH_CMD="git push"
if [ ! -z "$BRANCH" ]; then
    PUSH_CMD="$PUSH_CMD origin $BRANCH"
fi
if [ $FORCE_PUSH -eq 1 ]; then
    PUSH_CMD="$PUSH_CMD --force"
fi

if [ $DRY_RUN -eq 1 ]; then
    echo -e "${YELLOW}[试运行] $PUSH_CMD${NC}"
else
    if $PUSH_CMD; then
        echo -e "${GREEN}✓ 推送完成${NC}"
    else
        echo -e "${RED}✗ 推送失败${NC}"
        exit 1
    fi
fi

# 9. 显示结果
echo -e "\n${BLUE}========================================${NC}"
echo -e "${GREEN}✓ 操作成功完成！${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "提交信息: $COMMIT_MSG"
echo -e "最新提交:"
git log --oneline -3
echo -e "${BLUE}========================================${NC}"