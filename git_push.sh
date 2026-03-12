#!/bin/bash
# 文件名: git_push_only.sh
# 功能: 纯粹的推送脚本 - 只做 add, commit, push，没有任何拉取操作
# 适用: 当你想用本地代码强制覆盖远程时

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 显示帮助信息
show_help() {
    echo "用法: ./git_push_only.sh [选项]"
    echo ""
    echo "选项:"
    echo "  -p, --path        仓库路径 (必填)"
    echo "  -m, --message     提交信息 (必填)"
    echo "  -b, --branch      目标分支 (默认: main)"
    echo "  -f, --force       强制推送，覆盖远程"
    echo "  -e, --empty       允许空提交"
    echo "  -h, --help        显示帮助"
    echo ""
    echo "示例:"
    echo "  ./git_push_only.sh -p /nfs/xtjin/eval_pipline -m \"更新代码\""
    echo "  ./git_push_only.sh -p /nfs/xtjin/benchmark -m \"修复bug\" -b develop"
    echo "  ./git_push_only.sh -p /nfs/xtjin/eval_pipline -m \"强制覆盖\" -f"
    echo ""
    echo "注意: 此脚本不会执行任何 git pull/fetch/clone 操作"
    echo "      只做: git add → git commit → git push"
}

# 默认值
BRANCH="main"
FORCE_PUSH=0
ALLOW_EMPTY=0
REPO_PATH=""
COMMIT_MSG=""

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -p|--path)
            REPO_PATH="$2"
            shift 2
            ;;
        -m|--message)
            COMMIT_MSG="$2"
            shift 2
            ;;
        -b|--branch)
            BRANCH="$2"
            shift 2
            ;;
        -f|--force)
            FORCE_PUSH=1
            shift
            ;;
        -e|--empty)
            ALLOW_EMPTY=1
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo -e "${RED}错误: 未知选项 $1${NC}"
            show_help
            exit 1
            ;;
    esac
done

# 检查必填参数
if [ -z "$REPO_PATH" ]; then
    echo -e "${RED}错误: 必须指定仓库路径 (-p)${NC}"
    show_help
    exit 1
fi

if [ -z "$COMMIT_MSG" ]; then
    echo -e "${RED}错误: 必须提供提交信息 (-m)${NC}"
    show_help
    exit 1
fi

# 检查仓库是否存在
if [ ! -d "$REPO_PATH" ]; then
    echo -e "${RED}错误: 仓库路径不存在: $REPO_PATH${NC}"
    exit 1
fi

# 检查是否是 git 仓库
if [ ! -d "$REPO_PATH/.git" ]; then
    echo -e "${YELLOW}警告: $REPO_PATH 不是 Git 仓库${NC}"
    echo -e "是否要初始化 Git 仓库？(y/n): "
    read -r answer
    if [[ "$answer" == "y" || "$answer" == "Y" ]]; then
        cd "$REPO_PATH" || exit 1
        git init
        echo -e "${GREEN}✓ Git 仓库初始化完成${NC}"
    else
        echo -e "${RED}操作取消${NC}"
        exit 1
    fi
fi

# 切换到仓库目录
cd "$REPO_PATH" || { echo -e "${RED}错误: 无法进入目录 $REPO_PATH${NC}"; exit 1; }

# 显示基本信息
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Git 纯推送脚本${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "仓库: $(pwd)"
echo -e "当前分支: $(git branch --show-current 2>/dev/null || echo '无')"
echo -e "目标分支: $BRANCH"
echo -e "提交信息: $COMMIT_MSG"
echo -e "强制推送: $([ $FORCE_PUSH -eq 1 ] && echo '是' || echo '否')"
echo -e "空提交: $([ $ALLOW_EMPTY -eq 1 ] && echo '允许' || echo '不允许')"
echo -e "${BLUE}========================================${NC}\n"

# 1. 检查是否有远程仓库
echo -e "${YELLOW}[1/5] 检查远程仓库配置...${NC}"
if ! git remote -v | grep -q "origin"; then
    echo -e "${YELLOW}没有配置远程仓库，请添加:${NC}"
    echo "例如: git remote add origin git@github.com:username/repo.git"
    echo "或: git remote add origin https://github.com/username/repo.git"
    echo -e "\n请输入远程仓库地址 (直接回车跳过):"
    read -r remote_url
    if [ -n "$remote_url" ]; then
        git remote add origin "$remote_url"
        echo -e "${GREEN}✓ 远程仓库添加成功${NC}"
    else
        echo -e "${RED}✗ 没有远程仓库，无法推送${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}✓ 远程仓库已配置${NC}"
    git remote -v | head -1
fi

# 2. 添加所有文件
echo -e "\n${YELLOW}[2/5] 添加文件到暂存区...${NC}"
git add -A
if [ $? -ne 0 ]; then
    echo -e "${RED}✗ 添加文件失败${NC}"
    exit 1
fi

# 显示添加的文件数量
file_count=$(git status --porcelain | wc -l)
echo -e "${GREEN}✓ 已添加 $file_count 个文件${NC}"

# 3. 提交修改
echo -e "\n${YELLOW}[3/5] 提交修改...${NC}"
if git diff --cached --quiet; then
    if [ $ALLOW_EMPTY -eq 1 ]; then
        git commit --allow-empty -m "$COMMIT_MSG"
        echo -e "${GREEN}✓ 空提交成功${NC}"
    else
        echo -e "${YELLOW}⚠ 没有需要提交的修改${NC}"
        echo -e "提示: 使用 -e 参数允许空提交"
        exit 0
    fi
else
    git commit -m "$COMMIT_MSG"
    if [ $? -ne 0 ]; then
        echo -e "${RED}✗ 提交失败${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ 提交成功${NC}"
fi

# 4. 检查目标分支是否存在
echo -e "\n${YELLOW}[4/5] 检查分支...${NC}"
current_branch=$(git branch --show-current)
if [ -z "$current_branch" ]; then
    echo -e "${YELLOW}当前不在任何分支上，创建分支 $BRANCH${NC}"
    git checkout -b "$BRANCH"
elif [ "$current_branch" != "$BRANCH" ]; then
    echo -e "${YELLOW}当前分支 ($current_branch) 与目标分支 ($BRANCH) 不同${NC}"
    echo -e "是否要切换到 $BRANCH 分支？(y/n): "
    read -r answer
    if [[ "$answer" == "y" || "$answer" == "Y" ]]; then
        git checkout "$BRANCH" 2>/dev/null || git checkout -b "$BRANCH"
        echo -e "${GREEN}✓ 已切换到 $BRANCH 分支${NC}"
    else
        echo -e "${YELLOW}继续使用当前分支 $current_branch 推送${NC}"
        BRANCH="$current_branch"
    fi
fi

# 5. 推送到远程（纯推送，不拉取）
echo -e "\n${YELLOW}[5/5] 推送到远程仓库...${NC}"

# 构建推送命令
if [ $FORCE_PUSH -eq 1 ]; then
    PUSH_CMD="git push --force origin $BRANCH"
    echo -e "${RED}⚠ 警告: 使用强制推送将覆盖远程代码！${NC}"
else
    PUSH_CMD="git push origin $BRANCH"
fi

echo -e "执行: $PUSH_CMD"
echo -e "\n${YELLOW}确认推送？(y/n): ${NC}"
read -r confirm
if [[ "$confirm" == "y" || "$confirm" == "Y" ]]; then
    if $PUSH_CMD; then
        echo -e "\n${GREEN}✓ 推送成功！${NC}"
        echo -e "${BLUE}========================================${NC}"
        echo -e "最新提交:"
        git log --oneline -3 2>/dev/null
        echo -e "${BLUE}========================================${NC}"
    else
        echo -e "\n${RED}✗ 推送失败！${NC}"
        echo -e "${YELLOW}可能的原因:${NC}"
        echo -e "  1. 网络问题"
        echo -e "  2. 权限问题"
        echo -e "  3. 远程仓库有更新（可用 -f 强制覆盖）"
        exit 1
    fi
else
    echo -e "${YELLOW}操作取消${NC}"
    exit 0
fi