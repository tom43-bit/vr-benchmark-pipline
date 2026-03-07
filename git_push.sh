#!/bin/bash
# 文件名: git_push.sh
# 功能: 高级Git提交脚本，支持选择性提交，专为OVItest优化
# 注意: 无论有无修改都会尝试推送

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
    echo "  -m, --message    提交信息（如果没有修改，此信息用于空提交）"
    echo "  -f, --force      强制推送"
    echo "  -b, --branch     指定分支"
    echo "  -d, --dry-run    试运行（不实际提交）"
    echo "  -s, --set-upstream 设置上游分支"
    echo "  -a, --allow-empty 允许空提交"
    echo "  -h, --help       显示帮助"
    echo ""
    echo "示例:"
    echo "  ./git_push.sh -m \"提交信息\"                    # 普通提交"
    echo "  ./git_push.sh -m \"提交信息\" -b main            # 指定分支"
    echo "  ./git_push.sh -m \"首次提交\" -s                 # 首次提交设置上游"
    echo "  ./git_push.sh -m \"空提交\" -a                   # 即使无修改也提交"
}

# 解析参数
FORCE_PUSH=0
DRY_RUN=0
BRANCH=""
COMMIT_MSG=""
SET_UPSTREAM=0
ALLOW_EMPTY=0

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
        -s|--set-upstream)
            SET_UPSTREAM=1
            shift
            ;;
        -a|--allow-empty)
            ALLOW_EMPTY=1
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
echo -e "${GREEN}Git 高级提交脚本 - OVItest 专用${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "仓库: $(pwd)"
echo -e "分支: $(git branch --show-current)"
echo -e "时间: $(date)"
echo -e "${BLUE}========================================${NC}\n"

# 1. 检查状态
echo -e "${YELLOW}1. 检查工作区状态...${NC}"
git fetch --all
git status

HAS_CHANGES=$(git status --porcelain)
if [ -z "$HAS_CHANGES" ]; then
    echo -e "\n${YELLOW}⚠ 没有需要提交的修改${NC}"
    if [ $ALLOW_EMPTY -eq 0 ] && [ -z "$COMMIT_MSG" ]; then
        echo -e "${YELLOW}如果没有修改但想推送，请使用 -a 参数允许空提交${NC}"
        # 不退出，继续执行推送
    fi
fi

# 2. 显示修改详情（如果有）
if [ ! -z "$HAS_CHANGES" ]; then
    echo -e "\n${YELLOW}2. 修改详情:${NC}"
    echo -e "${CYAN}新增文件:${NC}"
    git ls-files --others --exclude-standard
    echo -e "\n${CYAN}修改文件:${NC}"
    git diff --name-only
    echo -e "\n${CYAN}删除文件:${NC}"
    git ls-files --deleted
fi

# 3. 获取提交信息
if [ -z "$COMMIT_MSG" ]; then
    if [ ! -z "$HAS_CHANGES" ] || [ $ALLOW_EMPTY -eq 1 ]; then
        echo -e "\n${YELLOW}3. 请输入提交信息:${NC}"
        read -p "提交信息: " COMMIT_MSG
        if [ -z "$COMMIT_MSG" ]; then
            COMMIT_MSG="提交给OVItest: $(date '+%Y-%m-%d %H:%M:%S')"
        fi
    fi
else
    echo -e "\n${YELLOW}3. 提交信息: ${COMMIT_MSG}${NC}"
fi

# 4. 添加文件（如果有修改）
if [ ! -z "$HAS_CHANGES" ]; then
    echo -e "\n${YELLOW}4. 添加文件...${NC}"
    git add .
    echo -e "${GREEN}✓ 添加完成${NC}"
else
    echo -e "\n${YELLOW}4. 没有文件需要添加${NC}"
fi

# 5. 提交（如果有修改或允许空提交）
echo -e "\n${YELLOW}5. 提交修改...${NC}"
if [ ! -z "$HAS_CHANGES" ] || [ $ALLOW_EMPTY -eq 1 ]; then
    if [ $DRY_RUN -eq 1 ]; then
        echo -e "${YELLOW}[试运行] git commit -m \"$COMMIT_MSG\"${NC}"
        if [ $ALLOW_EMPTY -eq 1 ] && [ -z "$HAS_CHANGES" ]; then
            echo -e "${YELLOW}[试运行] 这将是一个空提交${NC}"
        fi
    else
        if [ ! -z "$HAS_CHANGES" ]; then
            git commit -m "$COMMIT_MSG"
            echo -e "${GREEN}✓ 提交完成${NC}"
        elif [ $ALLOW_EMPTY -eq 1 ]; then
            git commit --allow-empty -m "$COMMIT_MSG"
            echo -e "${GREEN}✓ 空提交完成${NC}"
        fi
    fi
else
    echo -e "${YELLOW}⚠ 没有修改，跳过提交步骤${NC}"
fi

# 6. 检查并设置上游分支
echo -e "\n${YELLOW}6. 检查上游分支...${NC}"
CURRENT_BRANCH=$(git branch --show-current)

# 检查是否有上游分支
if ! git rev-parse --abbrev-ref --symbolic-full-name @{u} >/dev/null 2>&1; then
    echo -e "${YELLOW}当前分支 '$CURRENT_BRANCH' 没有设置上游分支${NC}"
    
    # 如果指定了分支，使用指定的
    if [ ! -z "$BRANCH" ]; then
        REMOTE_BRANCH="$BRANCH"
    else
        # 尝试常见的远程分支名
        for remote_branch in main master develop; do
            if git show-ref --verify --quiet refs/remotes/origin/$remote_branch; then
                REMOTE_BRANCH=$remote_branch
                break
            fi
        done
    fi
    
    if [ ! -z "$REMOTE_BRANCH" ]; then
        echo -e "${GREEN}找到远程分支: origin/$REMOTE_BRANCH${NC}"
        if [ $SET_UPSTREAM -eq 1 ] || [ $DRY_RUN -eq 0 ]; then
            echo -e "${YELLOW}设置上游分支: origin/$REMOTE_BRANCH${NC}"
            git branch --set-upstream-to=origin/$REMOTE_BRANCH $CURRENT_BRANCH
            echo -e "${GREEN}✓ 上游分支设置完成${NC}"
        else
            echo -e "${YELLOW}[试运行] 将设置上游分支: origin/$REMOTE_BRANCH${NC}"
        fi
    else
        echo -e "${RED}✗ 未找到远程分支，请手动指定: git branch --set-upstream-to=origin/<branch>${NC}"
        echo -e "${YELLOW}或者使用 -b 参数指定分支: ./git_push.sh -b main -m \"提交信息\"${NC}"
        exit 1
    fi
else
    UPSTREAM=$(git rev-parse --abbrev-ref --symbolic-full-name @{u})
    echo -e "${GREEN}✓ 上游分支已设置: $UPSTREAM${NC}"
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
        echo -e "${YELLOW}提示: 可能需要手动运行: git pull origin ${REMOTE_BRANCH:-main}${NC}"
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
    echo -e "${RED}警告: 使用强制推送可能会覆盖远程修改！${NC}"
fi

if [ $DRY_RUN -eq 1 ]; then
    echo -e "${YELLOW}[试运行] $PUSH_CMD${NC}"
else
    echo -e "${YELLOW}执行: $PUSH_CMD${NC}"
    if $PUSH_CMD; then
        echo -e "${GREEN}✓ 推送完成${NC}"
    else
        echo -e "${RED}✗ 推送失败${NC}"
        echo -e "${YELLOW}提示: 如果是因为没有上游分支，请添加 -s 参数:${NC}"
        echo -e "${YELLOW}  ./git_push.sh -m \"$COMMIT_MSG\" -s${NC}"
        exit 1
    fi
fi

# 9. 显示结果
echo -e "\n${BLUE}========================================${NC}"
echo -e "${GREEN}✓ 操作成功完成！${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "提交信息: ${COMMIT_MSG:-无}"
echo -e "分支: $CURRENT_BRANCH"
echo -e "远程分支: $(git rev-parse --abbrev-ref --symbolic-full-name @{u} 2>/dev/null || echo '未设置')"
echo -e "是否有修改: ${HAS_CHANGES:+是}${HAS_CHANGES:-否}"
echo -e "最新提交:"
git log --oneline -3 2>/dev/null || echo "无提交记录"
echo -e "${BLUE}========================================${NC}"